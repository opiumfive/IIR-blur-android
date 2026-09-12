package com.opiumfive.iirblurdemo;

import android.graphics.Bitmap;

public class Utils {

    static {
        System.loadLibrary("native-lib");
    }

    public static native void blurScalar(Bitmap dst, float strength);

    /** Fast up to sigma 128 (8-bit intermediate in the bitmap), precise beyond. */
    public static final int QUALITY_AUTOMATIC = 0;
    /** No extra image buffer; within one level of the reference for sigma up to 128. */
    public static final int QUALITY_FAST = 1;
    /** 16-bit intermediate; valid for any sigma. */
    public static final int QUALITY_PRECISE = 2;
    /**
     * Blurs a 2x/4x/8x box-downsampled copy and interpolates it back (from sigma 12).
     * Fastest for large sigma; a few levels off near the image borders and next to
     * hard high-contrast edges.
     */
    public static final int QUALITY_DRAFT = 3;

    /**
     * Blurs mutable software ARGB_8888 pixels in place; preserves alpha.
     * Sigma must be finite and non-negative (zero leaves the bitmap unchanged).
     * Uses NEON on arm64 and a portable fallback on other ABIs. Same as
     * {@code blurNeon(dst, strength, QUALITY_AUTOMATIC)}.
     */
    public static native void blurNeon(Bitmap dst, float strength);

    /** Like {@link #blurNeon(Bitmap, float)} with an explicit QUALITY_* level. */
    public static native void blurNeon(Bitmap dst, float strength, int quality);

    /**
     * How long the blur worker threads stay awake after a call, in milliseconds
     * (default 20). Phone CPUs drop their clocks within milliseconds of idling and
     * ramp up slower than a blur lasts, so a blur once per frame would otherwise run
     * several times slower than back to back. Raise it above the frame interval for
     * animations; set 0 for one-shot use where nothing should stay awake.
     */
    public static native void setBlurIdleSpin(int milliseconds);

    public static native void blurNeonFp16(Bitmap dst, float strength);

    public static native void blurBox(Bitmap dst, float strength);

}
