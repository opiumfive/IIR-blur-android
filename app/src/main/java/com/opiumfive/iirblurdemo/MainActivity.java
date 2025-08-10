package com.opiumfive.iirblurdemo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;


public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        System.loadLibrary("native-lib");

        ImageView res1 = findViewById(R.id.res1);
        ImageView res2 = findViewById(R.id.res2);
        ImageView res3 = findViewById(R.id.res3);
        ImageView res4 = findViewById(R.id.res4);
        ImageView res5 = findViewById(R.id.res5);
        TextView scalar = findViewById(R.id.scalar);
        TextView neon = findViewById(R.id.neon);
        TextView fp16 = findViewById(R.id.fp16);
        TextView fir = findViewById(R.id.fir);
        TextView gpu = findViewById(R.id.gpu);

        res1.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.back));
        res2.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.back));
        res3.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.back));
        res4.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.back));
        res5.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.back));

        float scale = 1f;

        findViewById(R.id.run).setOnClickListener((v) -> {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.back);
            bitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    (int) (bitmap.getWidth() * scale),
                    (int) (bitmap.getHeight() * scale),
                    true
            );

            long before = System.nanoTime();
            Utils.blurScalar(bitmap, 30f);
            long res = System.nanoTime() - before;
            res1.setImageBitmap(bitmap);
            scalar.setText("Scalar: " + res + " ns for bmp " + bitmap.getWidth() + "x" + bitmap.getHeight());

            Bitmap bitmap2 = BitmapFactory.decodeResource(getResources(), R.drawable.back);
            bitmap2 = Bitmap.createScaledBitmap(
                    bitmap2,
                    (int) (bitmap2.getWidth() * scale),
                    (int) (bitmap2.getHeight() * scale),
                    true
            );

            before = System.nanoTime();
            Utils.blurNeon(bitmap2, 30f);
            res = System.nanoTime() - before;
            res2.setImageBitmap(bitmap2);
            neon.setText("Neon: " + res + " ns");


            Bitmap bitmap3 = BitmapFactory.decodeResource(getResources(), R.drawable.back);
            bitmap3 = Bitmap.createScaledBitmap(
                    bitmap3,
                    (int) (bitmap3.getWidth() * scale),
                    (int) (bitmap3.getHeight() * scale),
                    true
            );

            before = System.nanoTime();
            Utils.blurNeonFp16(bitmap3, 30f);
            res = System.nanoTime() - before;
            res3.setImageBitmap(bitmap3);
            fp16.setText("Neon fp16: " + res + " ns");

            Bitmap bitmap4 = BitmapFactory.decodeResource(getResources(), R.drawable.back);
            bitmap4 = Bitmap.createScaledBitmap(
                    bitmap4,
                    (int) (bitmap4.getWidth() * scale),
                    (int) (bitmap4.getHeight() * scale),
                    true
            );

            before = System.nanoTime();
            Utils.blurBox(bitmap4, 30f);
            res = System.nanoTime() - before;
            res4.setImageBitmap(bitmap4);
            fir.setText("box neon: " + res + " ns");

            Bitmap bitmap5 = BitmapFactory.decodeResource(getResources(), R.drawable.back);
            bitmap5 = Bitmap.createScaledBitmap(
                    bitmap5,
                    (int) (bitmap5.getWidth() * scale),
                    (int) (bitmap5.getHeight() * scale),
                    true
            );

            before = System.nanoTime();
            bitmap5 = GpuBlurBitmap.blur(bitmap5, 15);
            //Utils.blurBox(bitmap4, 30f);
            res = System.nanoTime() - before;
            res5.setImageBitmap(bitmap5);
            gpu.setText("gpu: " + res + " ns");
        });


    }
}