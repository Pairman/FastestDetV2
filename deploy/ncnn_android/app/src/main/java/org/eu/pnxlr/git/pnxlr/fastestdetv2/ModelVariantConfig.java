package org.eu.pnxlr.git.pnxlr.fastestdetv2;

import android.content.Context;

final class ModelVariantConfig {
    static final String MODEL_VARIANT_1X = "1x";
    static final String MODEL_VARIANT_2X = "2x";

    private static final String PREFS_NAME = "fastestdetv2_prefs";
    private static final String PREF_MODEL_VARIANT = "model_variant";

    private ModelVariantConfig() {
    }

    static String normalize(String variant) {
        return MODEL_VARIANT_2X.equals(variant) ? MODEL_VARIANT_2X : MODEL_VARIANT_1X;
    }

    static String getCurrentVariant(Context context) {
        return normalize(context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .getString(PREF_MODEL_VARIANT, MODEL_VARIANT_1X));
    }

    static void setCurrentVariant(Context context, String variant) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(PREF_MODEL_VARIANT, normalize(variant))
                .apply();
    }

    static String getNextVariant(String currentVariant) {
        return MODEL_VARIANT_1X.equals(normalize(currentVariant)) ? MODEL_VARIANT_2X : MODEL_VARIANT_1X;
    }
}
