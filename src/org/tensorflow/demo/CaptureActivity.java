package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Handler;
import android.provider.MediaStore;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.IOException;
import java.util.List;

/**
 * Created by philips on 31/12/17.
 */

public class CaptureActivity extends Activity {
    private int PICK_IMAGE_REQUEST = 1;

    private static final String TAG = "CaptureActivity";
    private static final int PIXEL_WIDTH = 28;
    private static final String MODEL_FILE = "file:///android_asset/expert-emnist-graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";
    private static final int INPUT_SIZE = 28;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private Classifier classifier;
    private Runnable postInferenceCallback;
    private Handler handler;
    private long lastProcessingTimeMs;
    private TextView mResultText;
    private ImageView imageView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_capture);
        final Button button=(Button)findViewById(R.id.detect);

        mResultText = (TextView)findViewById(R.id.text_result);
        imageView = (ImageView) findViewById(R.id.image_holder);
        classifier =
                TensorFlowMnistImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Code here executes on main thread after user presses button
                // get Image from gallery and display
                Intent intent = new Intent();
// Show only images, no videos or anything else
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
// Always show the chooser (if there are multiple options available)
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {

            Uri uri = data.getData();

            try {
                Bitmap bitmap = Bitmap.createScaledBitmap(
                        MediaStore.Images.Media.getBitmap(getContentResolver(), uri)
                        ,28
                        ,28
                        ,true);
                // Log.d(TAG, String.valueOf(bitmap));
                int pixels[]=getPixelData(bitmap);
                final List<Classifier.Recognition> results = classifier.recognizeImages(pixels);

//        LOGGER.i("Results %s",results.get(0));
                if(results.size()>0){
                    mResultText.setText(results.get(0).getTitle());
                }
                else{
                    mResultText.setText("Unknown");
                }


                imageView.setImageBitmap(bitmap);
                Toast.makeText(getApplicationContext(),"Height"+bitmap.getHeight()+" Width"+bitmap.getWidth(),Toast.LENGTH_LONG).show();
                readyForNextImage();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static Bitmap Image(Bitmap src){
        int width = src.getWidth();
        int height = src.getHeight();
        // create output bitmap
        Bitmap bmOut = Bitmap.createBitmap(width, height, src.getConfig());
//        float sample[]=new float[width*height];
        // color information
        int A, R, G, B;int k=0;
        int pixel;
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                // get pixel color
                pixel = src.getPixel(x, y);
                A = Color.alpha(pixel);
                R = Color.red(pixel);
                G = Color.green(pixel);
                B = Color.blue(pixel);
                int gray = (int) (0.2989 * R + 0.5870 * G + 0.1140 * B);
                // use 128 as threshold, above -> white, below -> black
                if (gray > 128) {
                    gray = 255;
                }
                else{
                    gray = 0;
                }
//                sample[k++]=gray/255.0f;
                // set new pixel color to output bitmap
                bmOut.setPixel(x, y, Color.argb(A, gray, gray, gray));
            }
        }
        return bmOut;
    }

    protected synchronized void runInBackground(final Runnable r) {
//        LOGGER.i("Run in Background");
        if (handler != null) {
            handler.post(r);
        }

    }
    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }

    private int[] getPixelData(Bitmap bitmap){
        Matrix matrix = new Matrix();
        int width=bitmap.getWidth();
        int height=bitmap.getHeight();
        int pixels[]=new int[width*height];
        matrix.preScale(-1.0f, 1.0f);
        matrix.postRotate(270);
        Bitmap bOutput = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);

        bOutput.getPixels(pixels, 0, width, 0, 0, width, height);

        int[] retPixels = new int[pixels.length];
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixel
            int pix = pixels[i];
            int b = pix & 0xff;
            retPixels[i] = 0xff - b;
        }
        return retPixels;
    }
}
