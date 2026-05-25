package org.eu.pnxlr.git.pnxlr.fastestdetv2;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;


public class WelcomeActivity extends AppCompatActivity {
    private Button testButton;
    private Button benchButton;
    private Button modelToggleButton;
    private TextView modelTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcome);

        modelTextView = findViewById(R.id.tv_model);
        modelToggleButton = findViewById(R.id.btn_model_toggle);
        updateModelViews();

        modelToggleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String currentVariant = ModelVariantConfig.getCurrentVariant(WelcomeActivity.this);
                String nextVariant = ModelVariantConfig.getNextVariant(currentVariant);
                if (!FastestDetV2.init(getAssets(), nextVariant)) {
                    String message = "Model " + nextVariant + " load failed";
                    Toast.makeText(WelcomeActivity.this, message, Toast.LENGTH_SHORT).show();
                    return;
                }

                ModelVariantConfig.setCurrentVariant(WelcomeActivity.this, nextVariant);
                updateModelViews();
                Toast.makeText(WelcomeActivity.this, "Switched to model " + nextVariant, Toast.LENGTH_SHORT).show();
            }
        });

        testButton = findViewById(R.id.btn_test);
        testButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });

        benchButton = findViewById(R.id.btn_bench);
        benchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(WelcomeActivity.this, BenchActivity.class);
                WelcomeActivity.this.startActivity(intent);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateModelViews();
    }

    private void updateModelViews() {
        String currentVariant = ModelVariantConfig.getCurrentVariant(this);
        String nextVariant = ModelVariantConfig.getNextVariant(currentVariant);
        modelTextView.setText("Model: " + currentVariant);
        modelToggleButton.setText("Switch to " + nextVariant);
    }
}
