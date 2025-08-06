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
        TextView scalar = findViewById(R.id.scalar);
        TextView neon = findViewById(R.id.neon);

        res1.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.back));
        res2.setImageBitmap(BitmapFactory.decodeResource(getResources(), R.drawable.back));

        findViewById(R.id.run).setOnClickListener((v) -> {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.back);

            long before = System.nanoTime();
            Utils.blurScalar(bitmap, 30f);
            long res = System.nanoTime() - before;
            res1.setImageBitmap(bitmap);
            scalar.setText("Scalar: " + res + " ns for bmp " + bitmap.getWidth() + "x" + bitmap.getHeight());

            Bitmap bitmap2 = BitmapFactory.decodeResource(getResources(), R.drawable.back);
            before = System.nanoTime();
            Utils.blurNeon(bitmap2, 30f);
            res = System.nanoTime() - before;
            res2.setImageBitmap(bitmap2);
            neon.setText("Neon: " + res + " ns for bmp " + bitmap.getWidth() + "x" + bitmap.getHeight());
        });


    }
}