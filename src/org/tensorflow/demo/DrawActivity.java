package org.tensorflow.demo;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.demo.env.Logger;

import java.util.List;

/**
 * Created by philips on 25/12/17.
 */

public class DrawActivity extends Activity implements View.OnTouchListener {
    private static final String TAG = "DrawActivity";
    private static final int PIXEL_WIDTH = 28;
    private static final String MODEL_FILE = "file:///android_asset/expert-emnist-graph.pb";
    private static final int INPUT_SIZE = 28;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private Classifier classifier;
    private Runnable postInferenceCallback;
    private Handler handler;
    private long lastProcessingTimeMs;
    //  private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";
    private TextView mResultText;

    private float mLastX;
    private float mLastY;

    private DrawModel mModel;
    private DrawView mDrawView;

    private PointF mTmpPiont = new PointF();
    private static final Logger LOGGER = new Logger();

//    private DigitDetector mDetector = new DigitDetector();


    @SuppressWarnings("SuspiciousNameCombination")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw);

//        boolean ret = mDetector.setup(this);
//        if( !ret ) {
//            Log.i(TAG, "Detector setup failed");
//            return;
//        }

        mModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

        mDrawView = (DrawView) findViewById(R.id.view_draw);
        mDrawView.setModel(mModel);
        mDrawView.setOnTouchListener(this);

        View detectButton = findViewById(R.id.button_detect);
        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onDetectClicked();
            }
        });

        View clearButton = findViewById(R.id.button_clear);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClearClicked();
            }
        });

        mResultText = (TextView)findViewById(R.id.text_result);

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
    }

    @Override
    protected void onResume() {
        mDrawView.onResume();
        super.onResume();
    }

    @Override
    protected void onPause() {
        mDrawView.onPause();
        super.onPause();
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction() & MotionEvent.ACTION_MASK;

        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;

        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;

        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();
        mDrawView.calcPos(mLastX, mLastY, mTmpPiont);
        float lastConvX = mTmpPiont.x;
        float lastConvY = mTmpPiont.y;
        mModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        mDrawView.calcPos(x, y, mTmpPiont);
        float newConvX = mTmpPiont.x;
        float newConvY = mTmpPiont.y;
        mModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        mDrawView.invalidate();
    }

    private void processTouchUp() {
        mModel.endLine();
    }

    private void onDetectClicked() {
        final int pixels[] = mDrawView.getPixelData();
        final List<Classifier.Recognition> results = classifier.recognizeImages(pixels);
//        LOGGER.i("Results %s",results.get(0));
        if(results.size()>0){
            mResultText.setText(results.get(0).getTitle());
        }
        else{
            mResultText.setText("Unknown");
        }
        readyForNextImage();
//        Toast.makeText(getApplicationContext(),"Well, Hello there",Toast.LENGTH_SHORT).show();
//        runInBackground(
//                new Runnable() {
//                    @Override
//                    public void run() {
//                        final long startTime = SystemClock.uptimeMillis();
//                        final List<Classifier.Recognition> results = classifier.recognizeImage(pixels);
//                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//
////                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
////                        if (resultsView == null) {
////                            resultsView = (ResultsView) findViewById(R.id.results);
////                        }
////                        resultsView.setResults(results);
////                        requestRender();
////                        Toast.makeText(getApplicationContext(),"Well, Hello there",Toast.LENGTH_SHORT).show();
////                        readyForNextImage();
//                    }
//                });
//        int digit = mDetector.detectDigit(pixels);
//
//        Log.i(TAG, "digit =" + digit);
//
//        mResultText.setText("Detected = " + digit);
    }

    private void onClearClicked() {
        mModel.clear();
        mDrawView.reset();
        mDrawView.invalidate();

        mResultText.setText("");
    }

    protected synchronized void runInBackground(final Runnable r) {
        LOGGER.i("Run in Background");
        if (handler != null) {
            handler.post(r);
        }

    }
    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }
}
