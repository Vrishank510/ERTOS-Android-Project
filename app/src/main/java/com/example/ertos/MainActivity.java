package com.example.ertos;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private TextView xText, yText, zText, predictionText;
    private Sensor accelerometer;
    private SensorManager sensorManager;

    // TFLite model
    private Interpreter tflite;
    private float[] meanValues = {0f,  2f,  0f};
    private float[] stdValues = {0f, 2f, 0f};
    private String[] activityLabels = {"Sitting", "Walking", "Running"};

    private Handler predictionHandler = new Handler();
    private float lastX, lastY, lastZ;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeViews();
        initializeSensor();
        loadModel();
        startPredictionCycle();
    }

    private void initializeViews() {
        xText = findViewById(R.id.xText);
        yText = findViewById(R.id.yText);
        zText = findViewById(R.id.zText);
        predictionText = findViewById(R.id.predictionText);
        predictionText.setText("Activity: Unknown");
    }

    private void initializeSensor() {
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            if (accelerometer != null) {
                sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            } else {
                Toast.makeText(this, "No accelerometer sensor found", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void loadModel() {
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("activity_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void startPredictionCycle() {
        predictionHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                predictActivity(lastX, lastY, lastZ);
                predictionHandler.postDelayed(this, 2000); // 10 seconds
            }
        }, 2000); // Initial delay of 10 seconds
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            lastX = event.values[0];
            lastY = event.values[1];
            lastZ = event.values[2];

            // Update UI with raw values
            xText.setText(String.format(Locale.getDefault(), "X: %.2f", lastX));
            yText.setText(String.format(Locale.getDefault(), "Y: %.2f", lastY));
            zText.setText(String.format(Locale.getDefault(), "Z: %.2f", lastZ));
        }
    }

    private void predictActivity(float x, float y, float z) {
        // Normalize the input
        float[] input = new float[3];
        input[0] = (x - meanValues[0]) / stdValues[0];
        input[1] = (y - meanValues[1]) / stdValues[1];
        input[2] = (z - meanValues[2]) / stdValues[2];

        // Run inference
        float[][] output = new float[1][3];
        tflite.run(input, output);

        // Get predicted class
        int predictedClass = argmax(output[0]);
        String activity = activityLabels[predictedClass];

        // Update UI with prediction
        predictionText.setText("Activity: " + activity);
    }

    private int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not needed
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
        predictionHandler.removeCallbacksAndMessages(null);
    }

    @Override
    protected void onResume() {
        super.onResume();
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }
}