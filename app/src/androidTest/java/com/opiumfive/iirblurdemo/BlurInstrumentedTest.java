package com.opiumfive.iirblurdemo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.util.Arrays;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class BlurInstrumentedTest {
    @Test public void preservesConstantColourAndAlpha() {
        for (int w : new int[]{1, 3, 4, 5, 17, 129}) {
            Bitmap bitmap = Bitmap.createBitmap(w, 131, Bitmap.Config.ARGB_8888);
            bitmap.setPremultiplied(false);
            int[] input = new int[w * 131];
            for (int i = 0; i < input.length; ++i) input[i] = ((i % 256) << 24) | 0x00ff7f00;
            for (float sigma : new float[]{0, 0.5f, 30, 100, 1000}) {
                bitmap.setPixels(input, 0, w, 0, 0, w, 131);
                Utils.blurNeon(bitmap, sigma);
                int[] output = new int[input.length];
                bitmap.getPixels(output, 0, w, 0, 0, w, 131);
                assertArrayEquals(input, output);
            }
            bitmap.recycle();
        }
    }

    @Test public void rejectsInvalidArguments() {
        Bitmap bitmap = Bitmap.createBitmap(3, 5, Bitmap.Config.ARGB_8888);
        for (float sigma : new float[]{-1, Float.NaN, Float.POSITIVE_INFINITY}) {
            try { Utils.blurNeon(bitmap, sigma); fail("Accepted invalid sigma"); }
            catch (IllegalArgumentException expected) { }
        }
        try { Utils.blurNeon(null, 1); fail("Accepted null"); }
        catch (IllegalArgumentException expected) { }
        Bitmap immutable = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        try { Utils.blurNeon(immutable, 1); fail("Accepted immutable bitmap"); }
        catch (IllegalArgumentException expected) { }
        Bitmap rgb565 = Bitmap.createBitmap(3, 5, Bitmap.Config.RGB_565);
        try { Utils.blurNeon(rgb565, 1); fail("Accepted RGB565"); }
        catch (IllegalArgumentException expected) { }
        bitmap.recycle(); immutable.recycle(); rgb565.recycle();
    }

    @Test public void benchmarkRealBitmap() {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inMutable = true;
        options.inScaled = false;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap = BitmapFactory.decodeResource(
                InstrumentationRegistry.getInstrumentation().getTargetContext().getResources(),
                R.drawable.back, options);
        int w = bitmap.getWidth(), h = bitmap.getHeight();
        int[] pixels = new int[w * h];
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h);
        long[] times = new long[31];
        for (int i = -10; i < times.length; ++i) {
            bitmap.setPixels(pixels, 0, w, 0, 0, w, h);
            long start = System.nanoTime();
            Utils.blurNeon(bitmap, 30);
            long ns = System.nanoTime() - start;
            if (i >= 0) times[i] = ns;
        }
        Arrays.sort(times);
        Log.i("IIRBlurBenchmark", "Bitmap/JNI " + w + "x" + h + " sigma=30 median_ms="
                + times[15] / 1e6 + " p10_ms=" + times[3] / 1e6 + " p90_ms=" + times[27] / 1e6);
        int[] result = new int[w * h];
        bitmap.getPixels(result, 0, w, 0, 0, w, h);
        assertFalse("Blur did not change the image", Arrays.equals(pixels, result));
        bitmap.recycle();
    }
}
