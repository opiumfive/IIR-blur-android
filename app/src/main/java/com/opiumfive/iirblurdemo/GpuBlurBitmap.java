package com.opiumfive.iirblurdemo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.hardware.HardwareBuffer;
import android.media.Image;
import android.media.ImageReader;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.os.Build;
import android.view.Surface;

import androidx.annotation.RequiresApi;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * GPU Dual Kawase blur that returns:
 *  - API 29+: a GPU-backed HARDWARE Bitmap (suitable for ImageView, no CPU readback).
 *  - API <29: a regular ARGB_8888 Bitmap (fallback).
 *
 * Key points:
 *  - Headless EGL pbuffer context reused per-thread (ThreadLocal).
 *  - For HARDWARE path: draws final pass into an ImageReader-backed EGL window surface,
 *    acquires its HardwareBuffer and wraps it with Bitmap.wrapHardwareBuffer(..).
 *  - For CPU fallback path: single glReadPixels with flip+RB swap in shader to avoid CPU loops.
 *
 * Usage:
 *    Bitmap b = GpuBlurBitmap.blur(src, 25);            // HARDWARE on API 29+, else ARGB_8888
 *    Bitmap bCpu = GpuBlurBitmap.blurSoftware(src, 25); // force ARGB_8888
 *
 * Notes:
 *  - Returned HARDWARE bitmaps are immutable and require a hardware-accelerated Canvas to draw.
 *  - Radius clamped to [1..50].
 */
public final class GpuBlurBitmap {

    private GpuBlurBitmap() {}

    // Reuse one renderer per thread
    private static final ThreadLocal<Renderer> sRenderer = new ThreadLocal<>();

    /** Preferred API: returns HARDWARE Bitmap on API 29+, else ARGB_8888. */
    public static Bitmap blur(Bitmap src, int radius) {
        return blurInternal(src, radius, /*forceSoftware*/ false);
    }

    /** Force ARGB_8888 CPU Bitmap (e.g., if you need to access pixels). */
    public static Bitmap blurSoftware(Bitmap src, int radius) {
        return blurInternal(src, radius, /*forceSoftware*/ true);
    }

    private static Bitmap blurInternal(Bitmap src, int radius, boolean forceSoftware) {
        if (src == null) throw new IllegalArgumentException("src == null");
        if (src.isRecycled()) throw new IllegalArgumentException("src is recycled");

        radius = Math.max(1, Math.min(50, radius));
        Bitmap working = (src.getConfig() == Bitmap.Config.ARGB_8888) ? src
                : src.copy(Bitmap.Config.ARGB_8888, false);
        if (working == null) throw new RuntimeException("Failed to obtain ARGB_8888 bitmap");

        Renderer r = sRenderer.get();
        if (r == null) {
            r = new Renderer();
            sRenderer.set(r);
        }

        boolean wantHardware = !forceSoftware && Build.VERSION.SDK_INT >= 29;

        try {
            return r.runDualKawase(working, radius, wantHardware);
        } catch (RuntimeException ex) {
            // If GL context got lost, reset once and retry.
            r.release();
            sRenderer.remove();
            Renderer r2 = new Renderer();
            sRenderer.set(r2);
            return r2.runDualKawase(working, radius, wantHardware);
        } finally {
            if (working != src) working.recycle();
        }
    }

    /** Optional: release per-thread GL resources. */
    public static void shutdownForThisThread() {
        Renderer r = sRenderer.get();
        if (r != null) {
            r.release();
            sRenderer.remove();
        }
    }

    // ---------------------------------------------------------------------------------------------

    private static final class Renderer {

        // EGL
        private static final int EGL_OPENGL_ES3_BIT_KHR = 0x0040;
        private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
        private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
        private EGLSurface eglPbufferSurface = EGL14.EGL_NO_SURFACE; // kept current by default
        private boolean isEs3 = false;

        // Program / attrib / uniforms
        private int program = 0;
        private int aPosLoc = -1;
        private int aTexLoc = -1;
        private int uTexLoc = -1;
        private int uTexelLoc = -1;
        private int uOffsetLoc = -1;
        private int uFlipYLoc = -1;
        private int uSwapRBLoc = -1;

        // Fullscreen quad (triangle strip): pos(x,y), tex(s,t)
        private static final float[] FS_QUAD = new float[]{
                -1f, -1f, 0f, 0f,
                1f, -1f, 1f, 0f,
                -1f,  1f, 0f, 1f,
                1f,  1f, 1f, 1f
        };
        private final FloatBuffer quadBuffer;

        // Reusable readback buffer (CPU path)
        private ByteBuffer readback;

        // Shaders
        private static final String VERTEX =
                "attribute vec2 aPos;\n" +
                        "attribute vec2 aTex;\n" +
                        "varying vec2 vTex;\n" +
                        "uniform float uFlipY;\n" +
                        "void main(){\n" +
                        "  vTex = aTex;\n" +
                        "  if (uFlipY > 0.5) vTex.y = 1.0 - vTex.y;\n" +
                        "  gl_Position = vec4(aPos, 0.0, 1.0);\n" +
                        "}\n";

        private static final String FRAGMENT =
                "precision mediump float;\n" +
                        "uniform sampler2D uTex;\n" +
                        "uniform vec2 uTexel;\n" +
                        "uniform float uOffset;\n" +
                        "uniform float uSwapRB;\n" +
                        "varying vec2 vTex;\n" +
                        "void main(){\n" +
                        "  vec2 o = uTexel * uOffset;\n" +
                        "  vec4 sum = texture2D(uTex, vTex + o)\n" +
                        "           + texture2D(uTex, vTex - o)\n" +
                        "           + texture2D(uTex, vTex + vec2(o.x, -o.y))\n" +
                        "           + texture2D(uTex, vTex + vec2(-o.x, o.y));\n" +
                        "  vec4 col = sum * 0.25;\n" +
                        "  if (uSwapRB > 0.5) col = col.bgra;\n" +
                        "  gl_FragColor = col;\n" +
                        "}\n";

        Renderer() {
            quadBuffer = ByteBuffer.allocateDirect(FS_QUAD.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
            quadBuffer.put(FS_QUAD).position(0);
            initEGL();
            initGLObjects();
        }

        Bitmap runDualKawase(Bitmap src, int radius, boolean wantHardwareBitmap) {
            // GL max texture size
            int[] maxSizeArr = new int[1];
            GLES20.glGetIntegerv(GLES20.GL_MAX_TEXTURE_SIZE, maxSizeArr, 0);
            int maxTex = Math.max(2048, maxSizeArr[0]);

            int inW = src.getWidth();
            int inH = src.getHeight();

            Bitmap srcToProcess = src;
            Matrix restoreScale = null;

            if (inW > maxTex || inH > maxTex) {
                float scale = Math.min(maxTex / (float) inW, maxTex / (float) inH);
                int sw = Math.max(1, Math.round(inW * scale));
                int sh = Math.max(1, Math.round(inH * scale));
                srcToProcess = Bitmap.createScaledBitmap(src, sw, sh, true);
                restoreScale = new Matrix();
                restoreScale.postScale(inW / (float) sw, inH / (float) sh);
                inW = sw; inH = sh;
            }

            // Upload source
            GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1);
            int srcTex = createTextureFromBitmap(srcToProcess);

            // Pass mapping
            int passes = Math.max(1, Math.min(6, (int) Math.ceil(radius / 8f)));
            float baseOffset = Math.max(1f, radius / (float) passes * 0.6f);

            // Downsample
            List<Fbo> chain = new ArrayList<>(passes);
            int prevW = inW, prevH = inH;
            int inputTex = srcTex;
            int inputW = inW, inputH = inH;

            for (int i = 0; i < passes; i++) {
                int dw = Math.max(1, (prevW + 1) >> 1);
                int dh = Math.max(1, (prevH + 1) >> 1);
                Fbo f = new Fbo(dw, dh);
                f.init();
                chain.add(f);

                bindFbo(f);
                GLES20.glViewport(0, 0, dw, dh);
                drawKawase(inputTex, inputW, inputH, baseOffset + i * 0.6f, false, false);

                inputTex = f.texId;
                inputW = dw; inputH = dh;
                prevW = dw; prevH = dh;
            }

            // Upsample
            for (int i = passes - 2; i >= 0; i--) {
                Fbo target = chain.get(i);
                bindFbo(target);
                GLES20.glViewport(0, 0, target.w, target.h);
                drawKawase(inputTex, inputW, inputH, baseOffset + i * 0.6f, false, false);
                inputTex = target.texId;
                inputW = target.w; inputH = target.h;
            }

            Bitmap out;
            if (wantHardwareBitmap && Build.VERSION.SDK_INT >= 29) {
                // Final pass to ImageReader-backed window surface → Hardware Bitmap
                out = finalPassToHardwareBitmap(inputTex, inputW, inputH, inW, inH);
            } else {
                // CPU fallback: final pass into an FBO, then single readback (with flip+RB swap)
                Fbo outFbo = new Fbo(inW, inH);
                outFbo.init();
                bindFbo(outFbo);
                GLES20.glViewport(0, 0, inW, inH);
                drawKawase(inputTex, inputW, inputH, 1.0f, true, true); // flipY + swapRB for CPU copy
                out = readCurrentFboToBitmap(outFbo.w, outFbo.h);
                outFbo.release();
            }

            // Cleanup chain + source
            deleteTexture(srcTex);
            for (Fbo f : chain) f.release();

            // If we scaled down to fit maxTex, scale result back up to original size
            if (restoreScale != null) {
                if (out.getWidth() != src.getWidth() || out.getHeight() != src.getHeight()) {
                    Bitmap restored = Bitmap.createBitmap(src.getWidth(), src.getHeight(), out.getConfig() == Bitmap.Config.HARDWARE ? Bitmap.Config.ARGB_8888 : out.getConfig());
                    // If it's HARDWARE, draw into a temporary sw bitmap of same size then copy into restored
                    if (out.getConfig() == Bitmap.Config.HARDWARE) {
                        // Draw HARDWARE bitmap onto a software canvas at the right size
                        Bitmap tmp = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888);
                        new Canvas(tmp).drawBitmap(out, 0, 0, null);
                        Bitmap scaled = Bitmap.createBitmap(src.getWidth(), src.getHeight(), Bitmap.Config.ARGB_8888);
                        Canvas c = new Canvas(scaled);
                        c.drawBitmap(tmp, restoreScale, null);
                        tmp.recycle();
                        out = scaled;
                    } else {
                        Canvas c = new Canvas(restored);
                        c.drawBitmap(out, restoreScale, null);
                        out.recycle();
                        out = restored;
                    }
                }
            }

            return out;
        }

        // ---- HARDWARE final pass (API 29+) ----
        @RequiresApi(api = Build.VERSION_CODES.Q)
        private Bitmap finalPassToHardwareBitmap(int srcTex, int srcW, int srcH, int outW, int outH) {
            ImageReader reader = null;
            EGLSurface winSurface = EGL14.EGL_NO_SURFACE;
            Bitmap result;
            try {
                // Create an ImageReader whose buffers are GPU-color-output + GPU-sampled
                long usage = HardwareBuffer.USAGE_GPU_COLOR_OUTPUT | HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE;
                reader = ImageReader.newInstance(outW, outH, PixelFormat.RGBA_8888, /*maxImages*/1, usage);
                Surface surface = reader.getSurface();

                // Create a window surface for that Surface
                winSurface = EGL14.eglCreateWindowSurface(eglDisplay, eglConfigForWindow(), surface, new int[]{
                        EGL14.EGL_NONE
                }, 0);
                if (winSurface == null || winSurface == EGL14.EGL_NO_SURFACE) {
                    throw new RuntimeException("eglCreateWindowSurface failed");
                }

                // Draw final pass to the window surface
                if (!EGL14.eglMakeCurrent(eglDisplay, winSurface, winSurface, eglContext)) {
                    throw new RuntimeException("eglMakeCurrent(winSurface) failed");
                }
                // Draw to default framebuffer
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
                GLES20.glViewport(0, 0, outW, outH);
                // flipY=true so the buffer is top-down for UI; swapRB=false (we're staying on GPU)
                drawKawase(srcTex, srcW, srcH, 1.0f, true, false);

                // Present into ImageReader
                EGL14.eglSwapBuffers(eglDisplay, winSurface);

                // Synchronously acquire the image and wrap its HardwareBuffer
                Image image = null;
                try {
                    // A short, bounded poll; on modern devices this is usually ready immediately.
                    for (int i = 0; i < 10 && (image = reader.acquireLatestImage()) == null; i++) {
                        Thread.yield();
                    }
                    if (image == null) {
                        // Last resort (can block briefly)
                        image = reader.acquireNextImage();
                    }
                    if (image == null) {
                        throw new RuntimeException("ImageReader returned null Image");
                    }
                    HardwareBuffer hb = image.getHardwareBuffer(); // API 29+
                    if (hb == null) throw new RuntimeException("No HardwareBuffer from ImageReader");
                    Bitmap hw = Bitmap.wrapHardwareBuffer(hb, null); // HARDWARE Bitmap
                    // We can close our local ref; Bitmap keeps its own ref.
                    hb.close();
                    result = hw;
                } finally {
                    if (image != null) image.close();
                }
            } finally {
                // Restore pbuffer as current
                if (eglPbufferSurface != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglMakeCurrent(eglDisplay, eglPbufferSurface, eglPbufferSurface, eglContext);
                }
                if (winSurface != null && winSurface != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglDestroySurface(eglDisplay, winSurface);
                }
                if (reader != null) {
                    reader.close();
                }
            }
            return result;
        }

        // ---- CPU readback path ----
        private Bitmap readCurrentFboToBitmap(int w, int h) {
            int cap = w * h * 4;
            if (readback == null || readback.capacity() < cap) {
                readback = ByteBuffer.allocateDirect(cap).order(ByteOrder.nativeOrder());
            }
            readback.position(0);
            readback.limit(cap);

            GLES20.glPixelStorei(GLES20.GL_PACK_ALIGNMENT, 1);
            GLES20.glReadPixels(0, 0, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, readback);

            Bitmap out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            readback.position(0);
            out.copyPixelsFromBuffer(readback);
            return out;
        }

        // --------------------------------- EGL ---------------------------------

        private EGLConfig cachedWindowConfig; // config that supports WINDOW_BIT

        private void initEGL() {
            eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
            if (eglDisplay == EGL14.EGL_NO_DISPLAY) throw new RuntimeException("eglGetDisplay failed");

            int[] version = new int[2];
            if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
                throw new RuntimeException("eglInitialize failed");
            }

            // Choose a config that supports both PBUFFER and WINDOW
            EGLConfig config = chooseConfig(true);
            if (config == null) {
                config = chooseConfig(false);
                if (config == null) throw new RuntimeException("No suitable EGLConfig");
            }

            int[] ctxAttribs = {
                    EGL14.EGL_CONTEXT_CLIENT_VERSION, isEs3 ? 3 : 2,
                    EGL14.EGL_NONE
            };
            eglContext = EGL14.eglCreateContext(eglDisplay, config, EGL14.EGL_NO_CONTEXT, ctxAttribs, 0);
            if (eglContext == null || eglContext == EGL14.EGL_NO_CONTEXT) {
                throw new RuntimeException("eglCreateContext failed");
            }

            // Tiny pbuffer to keep context current
            int[] pbufferAttribs = {
                    EGL14.EGL_WIDTH, 1,
                    EGL14.EGL_HEIGHT, 1,
                    EGL14.EGL_NONE
            };
            eglPbufferSurface = EGL14.eglCreatePbufferSurface(eglDisplay, config, pbufferAttribs, 0);
            if (eglPbufferSurface == null || eglPbufferSurface == EGL14.EGL_NO_SURFACE) {
                throw new RuntimeException("eglCreatePbufferSurface failed");
            }

            if (!EGL14.eglMakeCurrent(eglDisplay, eglPbufferSurface, eglPbufferSurface, eglContext)) {
                throw new RuntimeException("eglMakeCurrent pbuffer failed");
            }

            // Remember a window-capable config (same as above)
            cachedWindowConfig = config;
        }

        private EGLConfig eglConfigForWindow() {
            return cachedWindowConfig != null ? cachedWindowConfig : chooseConfig(isEs3);
        }

        private EGLConfig chooseConfig(boolean tryEs3) {
            int renderable = tryEs3 ? EGL_OPENGL_ES3_BIT_KHR : EGL14.EGL_OPENGL_ES2_BIT;
            int[] attribList = {
                    EGL14.EGL_RED_SIZE, 8,
                    EGL14.EGL_GREEN_SIZE, 8,
                    EGL14.EGL_BLUE_SIZE, 8,
                    EGL14.EGL_ALPHA_SIZE, 8,
                    EGL14.EGL_RENDERABLE_TYPE, renderable,
                    EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT | EGL14.EGL_WINDOW_BIT,
                    EGL14.EGL_NONE
            };
            EGLConfig[] configs = new EGLConfig[1];
            int[] numConfig = new int[1];
            if (!EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, 1, numConfig, 0) || numConfig[0] <= 0) {
                isEs3 = false;
                return null;
            }
            isEs3 = tryEs3;
            return configs[0];
        }

        private void releaseEGL() {
            if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
                if (eglPbufferSurface != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglDestroySurface(eglDisplay, eglPbufferSurface);
                    eglPbufferSurface = EGL14.EGL_NO_SURFACE;
                }
                if (eglContext != EGL14.EGL_NO_CONTEXT) {
                    EGL14.eglDestroyContext(eglDisplay, eglContext);
                    eglContext = EGL14.EGL_NO_CONTEXT;
                }
                EGL14.eglTerminate(eglDisplay);
                eglDisplay = EGL14.EGL_NO_DISPLAY;
            }
        }

        // --------------------------------- GL ----------------------------------

        private void initGLObjects() {
            int vs = compileShader(GLES20.GL_VERTEX_SHADER, VERTEX);
            int fs = compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT);
            program = GLES20.glCreateProgram();
            GLES20.glAttachShader(program, vs);
            GLES20.glAttachShader(program, fs);
            GLES20.glLinkProgram(program);
            int[] link = new int[1];
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, link, 0);
            if (link[0] != GLES20.GL_TRUE) {
                String log = GLES20.glGetProgramInfoLog(program);
                GLES20.glDeleteProgram(program);
                GLES20.glDeleteShader(vs);
                GLES20.glDeleteShader(fs);
                throw new RuntimeException("Program link failed: " + log);
            }
            GLES20.glDeleteShader(vs);
            GLES20.glDeleteShader(fs);

            aPosLoc = GLES20.glGetAttribLocation(program, "aPos");
            aTexLoc = GLES20.glGetAttribLocation(program, "aTex");
            uTexLoc = GLES20.glGetUniformLocation(program, "uTex");
            uTexelLoc = GLES20.glGetUniformLocation(program, "uTexel");
            uOffsetLoc = GLES20.glGetUniformLocation(program, "uOffset");
            uFlipYLoc = GLES20.glGetUniformLocation(program, "uFlipY");
            uSwapRBLoc = GLES20.glGetUniformLocation(program, "uSwapRB");

            GLES20.glDisable(GLES20.GL_DEPTH_TEST);
            GLES20.glDisable(GLES20.GL_CULL_FACE);
        }

        private static int compileShader(int type, String src) {
            int id = GLES20.glCreateShader(type);
            GLES20.glShaderSource(id, src);
            GLES20.glCompileShader(id);
            int[] compiled = new int[1];
            GLES20.glGetShaderiv(id, GLES20.GL_COMPILE_STATUS, compiled, 0);
            if (compiled[0] == 0) {
                String log = GLES20.glGetShaderInfoLog(id);
                GLES20.glDeleteShader(id);
                throw new RuntimeException((type == GLES20.GL_VERTEX_SHADER ? "Vertex" : "Fragment")
                        + " shader compile failed: " + log);
            }
            return id;
        }

        private int createTextureFromBitmap(Bitmap bmp) {
            int[] ids = new int[1];
            GLES20.glGenTextures(1, ids, 0);
            int id = ids[0];
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
            return id;
        }

        private void deleteTexture(int id) {
            if (id != 0) {
                int[] a = new int[]{id};
                GLES20.glDeleteTextures(1, a, 0);
            }
        }

        private void bindFbo(Fbo f) {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, f.fboId);
        }

        private void drawKawase(int tex, int srcW, int srcH, float offset, boolean flipY, boolean swapRB) {
            GLES20.glUseProgram(program);

            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex);
            GLES20.glUniform1i(uTexLoc, 0);

            GLES20.glUniform2f(uTexelLoc, 1f / srcW, 1f / srcH);
            GLES20.glUniform1f(uOffsetLoc, offset);
            GLES20.glUniform1f(uFlipYLoc, flipY ? 1f : 0f);
            GLES20.glUniform1f(uSwapRBLoc, swapRB ? 1f : 0f);

            quadBuffer.position(0);
            GLES20.glEnableVertexAttribArray(aPosLoc);
            GLES20.glVertexAttribPointer(aPosLoc, 2, GLES20.GL_FLOAT, false, 4 * 4, quadBuffer);

            quadBuffer.position(2);
            GLES20.glEnableVertexAttribArray(aTexLoc);
            GLES20.glVertexAttribPointer(aTexLoc, 2, GLES20.GL_FLOAT, false, 4 * 4, quadBuffer);

            GLES20.glDisable(GLES20.GL_BLEND);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

            GLES20.glDisableVertexAttribArray(aPosLoc);
            GLES20.glDisableVertexAttribArray(aTexLoc);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        }

        void release() {
            if (program != 0) {
                GLES20.glDeleteProgram(program);
                program = 0;
            }
            releaseEGL();
            readback = null;
        }

        // ------------------------------- FBO wrapper ---------------------------

        private static final class Fbo {
            final int w, h;
            int fboId = 0;
            int texId = 0;

            Fbo(int w, int h) { this.w = w; this.h = h; }

            void init() {
                int[] ids = new int[1];

                // Texture
                GLES20.glGenTextures(1, ids, 0);
                texId = ids[0];
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
                GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, w, h, 0,
                        GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);

                // FBO
                GLES20.glGenFramebuffers(1, ids, 0);
                fboId = ids[0];
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId);
                GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                        GLES20.GL_TEXTURE_2D, texId, 0);

                int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
                if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
                    throw new RuntimeException("FBO incomplete: 0x" + Integer.toHexString(status) + " (" + w + "x" + h + ")");
                }

                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
            }

            void release() {
                if (fboId != 0) {
                    int[] a = new int[]{fboId};
                    GLES20.glDeleteFramebuffers(1, a, 0);
                    fboId = 0;
                }
                if (texId != 0) {
                    int[] a = new int[]{texId};
                    GLES20.glDeleteTextures(1, a, 0);
                    texId = 0;
                }
            }
        }
    }
}
