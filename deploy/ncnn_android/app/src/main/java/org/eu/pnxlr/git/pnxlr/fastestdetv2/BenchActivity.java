package org.eu.pnxlr.git.pnxlr.fastestdetv2;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.Toast;
import android.widget.TextView;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BenchActivity extends AppCompatActivity {
    private TextView tvBench;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_bench);

        tvBench = findViewById(R.id.tv_bench);
        String currentModelVariant = ModelVariantConfig.getCurrentVariant(this);
        if (!FastestDetV2.init(getAssets(), currentModelVariant)) {
            tvBench.setText("Model " + currentModelVariant + " load failed");
            Toast.makeText(this, "Model " + currentModelVariant + " load failed", Toast.LENGTH_SHORT).show();
            return;
        }
        tvBench.setText("Running benchmark...");
        executor.execute(new Runnable() {
            @Override
            public void run() {
                final String result = FastestDetV2.bench(300);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (!isFinishing() && !isDestroyed()) {
                            tvBench.setText(result);
                        }
                    }
                });
            }
        });
    }

    @Override
    protected void onDestroy() {
        executor.shutdown();
        super.onDestroy();
    }
}
