package com.opiumfive.iirblurdemo;

import android.graphics.Bitmap;

public class Utils {

    static {
        System.loadLibrary("native-lib");
    }

    public static native void blurScalar(Bitmap dst, float strength);

    /**
     * Blurs mutable software ARGB_8888 pixels in place; preserves alpha.
     * Sigma must be finite and non-negative (zero leaves the bitmap unchanged).
     * Uses NEON on arm64 and a portable fallback on other ABIs.
     */
    public static native void blurNeon(Bitmap dst, float strength);

    public static native void blurNeonFp16(Bitmap dst, float strength);

    public static native void blurBox(Bitmap dst, float strength);

}
