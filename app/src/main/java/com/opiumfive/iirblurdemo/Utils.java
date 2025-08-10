package com.opiumfive.iirblurdemo;

import android.graphics.Bitmap;

public class Utils {

    public static native void blurScalar(Bitmap dst, float strength);

    public static native void blurNeon(Bitmap dst, float strength);

    public static native void blurNeonFp16(Bitmap dst, float strength);

    public static native void blurBox(Bitmap dst, float strength);

}
