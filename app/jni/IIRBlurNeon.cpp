#include "IIRBlur.h"

#include <android/bitmap.h>
#include <cmath>
#include <jni.h>

namespace {
void throwJava(JNIEnv *env, const char *type, const char *message) {
    if (env->ExceptionCheck())
        return;
    jclass cls = env->FindClass(type);
    if (cls)
        env->ThrowNew(cls, message);
}
} // namespace

extern "C" JNIEXPORT void JNICALL Java_com_opiumfive_iirblurdemo_Utils_blurNeon(JNIEnv *env, jclass,
                                                                                jobject bitmap,
                                                                                jfloat sigma) {
    if (!bitmap || !std::isfinite(sigma) || sigma < 0) {
        throwJava(env, "java/lang/IllegalArgumentException",
                  "A bitmap and a finite, non-negative sigma are required");
        return;
    }
    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || !info.width || !info.height) {
        throwJava(env, "java/lang/IllegalArgumentException",
                  "Expected a software ARGB_8888 bitmap");
        return;
    }
    // lockPixels alone does not enforce the Java Bitmap mutability contract.
    jclass cls = env->GetObjectClass(bitmap);
    jmethodID isMutable = env->GetMethodID(cls, "isMutable", "()Z");
    if (!isMutable || !env->CallBooleanMethod(bitmap, isMutable)) {
        throwJava(env, "java/lang/IllegalArgumentException", "The bitmap must be mutable");
        return;
    }
    if (sigma == 0)
        return;
    void *pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throwJava(env, "java/lang/IllegalArgumentException", "Unable to lock bitmap pixels");
        return;
    }
    bool ok =
        iirblur::blur(static_cast<uint8_t *>(pixels), info.width, info.height, info.stride, sigma);
    AndroidBitmap_unlockPixels(env, bitmap);
    if (!ok)
        throwJava(env, "java/lang/OutOfMemoryError", "Unable to allocate blur workspace");
}
