package org.eu.pnxlr.git.pnxlr.fastestdetv2;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.UseCase;
import androidx.core.app.ActivityCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Size;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CAMERA = 1;
    private static final String[] PERMISSIONS_CAMERA = {Manifest.permission.CAMERA};
    private static final CameraX.LensFacing CAMERA_FACING = CameraX.LensFacing.BACK;
    private static final String STATE_MODEL_VARIANT = "state_model_variant";

    private ImageView resultImageView;
    private TextureView viewFinder;
    private SeekBar nmsSeekBar;
    private SeekBar thresholdSeekBar;
    private TextView thresholdTextview;
    private TextView tvInfo;

    private double threshold = 0.65f;
    private double nmsThreshold = 0.45f;
    private double totalFps = 0.0;
    private int fpsCount = 0;
    private String currentModelVariant = ModelVariantConfig.MODEL_VARIANT_1X;

    private final ExecutorService detectExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean detecting = new AtomicBoolean(false);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        currentModelVariant = savedInstanceState != null
                ? ModelVariantConfig.normalize(savedInstanceState.getString(
                STATE_MODEL_VARIANT, ModelVariantConfig.MODEL_VARIANT_1X))
                : ModelVariantConfig.getCurrentVariant(this);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, PERMISSIONS_CAMERA, REQUEST_CAMERA);
            return;
        }

        initDetectorPage();
    }

    private void initDetectorPage() {
        resultImageView = findViewById(R.id.imageView);
        viewFinder = findViewById(R.id.view_finder);
        thresholdTextview = findViewById(R.id.valTxtView);
        tvInfo = findViewById(R.id.tv_info);
        nmsSeekBar = findViewById(R.id.nms_seek);
        thresholdSeekBar = findViewById(R.id.threshold_seek);

        nmsSeekBar.setProgress((int) (nmsThreshold * 100));
        thresholdSeekBar.setProgress((int) (threshold * 100));
        updateThresholdText();

        nmsSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                nmsThreshold = progress / 100.f;
                updateThresholdText();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                threshold = progress / 100.f;
                updateThresholdText();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        viewFinder.addOnLayoutChangeListener(new View.OnLayoutChangeListener() {
            @Override
            public void onLayoutChange(View v, int left, int top, int right, int bottom,
                                       int oldLeft, int oldTop, int oldRight, int oldBottom) {
                updateTransform();
            }
        });

        viewFinder.post(new Runnable() {
            @Override
            public void run() {
                initCurrentModel();
                startCamera();
            }
        });
    }

    private void initCurrentModel() {
        currentModelVariant = ModelVariantConfig.getCurrentVariant(this);
        boolean loaded = FastestDetV2.init(getAssets(), currentModelVariant);
        if (!loaded) {
            String message = "Model " + currentModelVariant + " load failed";
            tvInfo.setText(message);
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
            return;
        }

        totalFps = 0.0;
        fpsCount = 0;
    }

    private void updateThresholdText() {
        thresholdTextview.setText(String.format(Locale.ENGLISH,
                "Conf Thres: %.2f    IoU Thres: %.2f",
                threshold,
                nmsThreshold));
    }

    private void updateTransform() {
        if (viewFinder.getDisplay() == null) {
            return;
        }

        Matrix matrix = new Matrix();
        float centerX = viewFinder.getWidth() / 2f;
        float centerY = viewFinder.getHeight() / 2f;
        float[] rotations = {0, 90, 180, 270};
        float rotationDegrees = rotations[viewFinder.getDisplay().getRotation()];
        matrix.postRotate(-rotationDegrees, centerX, centerY);
        viewFinder.setTransform(matrix);
    }

    private void startCamera() {
        CameraX.unbindAll();

        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CAMERA_FACING)
                .setTargetResolution(new Size(320, 320))
                .build();

        Preview preview = new Preview(previewConfig);
        preview.setOnPreviewOutputUpdateListener(new Preview.OnPreviewOutputUpdateListener() {
            @Override
            public void onUpdated(Preview.PreviewOutput output) {
                ViewGroup parent = (ViewGroup) viewFinder.getParent();
                parent.removeView(viewFinder);
                parent.addView(viewFinder, 0);
                viewFinder.setSurfaceTexture(output.getSurfaceTexture());
                updateTransform();
            }
        });

        CameraX.bindToLifecycle((LifecycleOwner) this, preview, createAnalyzerUseCase());
    }

    private UseCase createAnalyzerUseCase() {
        ImageAnalysisConfig config = new ImageAnalysisConfig.Builder()
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .setTargetResolution(new Size(320, 320))
                .build();

        ImageAnalysis analysis = new ImageAnalysis(config);
        analysis.setAnalyzer(new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(ImageProxy image, int rotationDegrees) {
                analyzeImage(image, rotationDegrees);
            }
        });
        return analysis;
    }

    private void analyzeImage(ImageProxy image, int rotationDegrees) {
        if (!detecting.compareAndSet(false, true)) {
            return;
        }

        long startTime = System.currentTimeMillis();
        try {
            Bitmap sourceBitmap = imageToBitmap(image);
            if (sourceBitmap == null) {
                return;
            }

            Bitmap bitmap = rotateBitmap(sourceBitmap, rotationDegrees);
            Box[] result = FastestDetV2.detect(bitmap, threshold, nmsThreshold);
            Bitmap resultBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
            resultBitmap.eraseColor(Color.TRANSPARENT);
            resultBitmap = drawBoxRects(resultBitmap, result);

            final int width = bitmap.getWidth();
            final int height = bitmap.getHeight();
            final long durationMs = System.currentTimeMillis() - startTime;
            final float fps = durationMs > 0 ? (float) (1000.0 / durationMs) : 0.f;
            totalFps = (totalFps == 0) ? fps : (totalFps + fps);
            fpsCount++;
            final float avgFps = fpsCount > 0 ? (float) (totalFps / fpsCount) : 0.f;
            final Bitmap finalBitmap = resultBitmap;
            final String infoText = String.format(
                    Locale.CHINESE,
                    "Frame: %dx%d\nTime: %.3f s\nFPS: %.3f\nAVG_FPS: %.3f",
                    height,
                    width,
                    durationMs / 1000.0,
                    fps,
                    avgFps
            );

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (isFinishing() || isDestroyed()) {
                        return;
                    }
                    resultImageView.setImageBitmap(finalBitmap);
                    tvInfo.setText(infoText);
                }
            });
        } catch (RuntimeException e) {
            final String message = e.getClass().getSimpleName() + ": " + String.valueOf(e.getMessage());
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (isFinishing() || isDestroyed()) {
                        return;
                    }
                    tvInfo.setText(message);
                    Toast.makeText(MainActivity.this, "Inference failed", Toast.LENGTH_SHORT).show();
                }
            });
        } finally {
            detecting.set(false);
        }
    }

    private Bitmap rotateBitmap(Bitmap sourceBitmap, int rotationDegrees) {
        if (rotationDegrees == 0) {
            return sourceBitmap;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        return Bitmap.createBitmap(sourceBitmap, 0, 0, sourceBitmap.getWidth(), sourceBitmap.getHeight(), matrix, false);
    }

    private Bitmap imageToBitmap(ImageProxy image) {
        byte[] nv21 = imageToNV21(image);
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 95, out);
        byte[] imageBytes = out.toByteArray();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length, options);
    }

    private byte[] imageToNV21(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ImageProxy.PlaneProxy y = planes[0];
        ImageProxy.PlaneProxy u = planes[1];
        ImageProxy.PlaneProxy v = planes[2];
        ByteBuffer yBuffer = y.getBuffer();
        ByteBuffer uBuffer = u.getBuffer();
        ByteBuffer vBuffer = v.getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);
        return nv21;
    }

    private Bitmap drawBoxRects(Bitmap mutableBitmap, Box[] results) {
        if (results == null || results.length == 0) {
            return mutableBitmap;
        }

        Canvas canvas = new Canvas(mutableBitmap);
        Paint boxPaint = new Paint();
        boxPaint.setAlpha(200);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4 * mutableBitmap.getWidth() / 800.0f);
        boxPaint.setTextSize(40 * mutableBitmap.getWidth() / 800.0f);

        for (Box box : results) {
            boxPaint.setColor(box.getColor());
            boxPaint.setStyle(Paint.Style.FILL);
            canvas.drawText(
                    box.getLabel() + String.format(Locale.CHINESE, " %.3f", box.getScore()),
                    box.x0 + 3,
                    box.y0 + 40 * mutableBitmap.getWidth() / 1000.0f,
                    boxPaint
            );
            boxPaint.setStyle(Paint.Style.STROKE);
            canvas.drawRect(box.getRect(), boxPaint);
        }

        return mutableBitmap;
    }

    @Override
    protected void onDestroy() {
        CameraX.unbindAll();
        detectExecutor.shutdown();
        super.onDestroy();
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putString(STATE_MODEL_VARIANT, currentModelVariant);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_CAMERA) {
            return;
        }

        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera Permission!", Toast.LENGTH_SHORT).show();
                finish();
                return;
            }
        }

        initDetectorPage();
    }
}
