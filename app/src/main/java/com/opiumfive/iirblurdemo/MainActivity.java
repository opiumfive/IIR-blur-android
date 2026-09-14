package com.opiumfive.iirblurdemo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    interface Blur {
        void run(Bitmap bitmap);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        System.loadLibrary("native-lib");

        int[] images = {R.id.res1, R.id.res2, R.id.res3, R.id.res4, R.id.res5, R.id.res6, R.id.res7};
        for (int id : images)
            ((ImageView) findViewById(id)).setImageBitmap(sourceBitmap());

        findViewById(R.id.run).setOnClickListener((v) -> {
            Bitmap probe = sourceBitmap();
            ((TextView) findViewById(R.id.size)).setText(
                    "bitmap " + probe.getWidth() + "x" + probe.getHeight() + ", sigma 30");
            probe.recycle();
            measure(R.id.res1, R.id.scalar, "original scalar", b -> Utils.blurScalar(b, 30f));
            measure(R.id.res2, R.id.neon, "IIR fast", b -> Utils.blurNeon(b, 30f, Utils.QUALITY_FAST));
            measure(R.id.res3, R.id.precise, "IIR precise", b -> Utils.blurNeon(b, 30f, Utils.QUALITY_PRECISE));
            measure(R.id.res4, R.id.draft, "IIR draft", b -> Utils.blurNeon(b, 30f, Utils.QUALITY_DRAFT));
            measure(R.id.res5, R.id.fp16, "original fp16", b -> Utils.blurNeonFp16(b, 30f));
            measure(R.id.res6, R.id.fir, "box neon", b -> Utils.blurBox(b, 30f));
            ImageView gpuView = findViewById(R.id.res7);
            Bitmap gpuBitmap = sourceBitmap();
            long before = System.nanoTime();
            gpuBitmap = GpuBlurBitmap.blur(gpuBitmap, 15);
            long ns = System.nanoTime() - before;
            gpuView.setImageBitmap(gpuBitmap);
            ((TextView) findViewById(R.id.gpu)).setText(String.format(Locale.US, "gpu: %.1f ms", ns / 1e6));
        });
    }

    /** Times a cold call and a second, warm call on fresh copies of the source. */
    private void measure(int imageId, int labelId, String name, Blur blur) {
        Bitmap first = sourceBitmap();
        long before = System.nanoTime();
        blur.run(first);
        long cold = System.nanoTime() - before;
        Bitmap second = sourceBitmap();
        before = System.nanoTime();
        blur.run(second);
        long warm = System.nanoTime() - before;
        first.recycle();
        ((ImageView) findViewById(imageId)).setImageBitmap(second);
        ((TextView) findViewById(labelId)).setText(
                String.format(Locale.US, "%s: %.1f ms (first %.1f)", name, warm / 1e6, cold / 1e6));
    }

    private Bitmap sourceBitmap() {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inMutable = true;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeResource(getResources(), R.drawable.back, options);
    }
}
