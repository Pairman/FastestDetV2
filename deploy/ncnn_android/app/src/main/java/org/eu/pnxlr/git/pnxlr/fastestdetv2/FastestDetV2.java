package org.eu.pnxlr.git.pnxlr.fastestdetv2;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class FastestDetV2 {
    static {
        System.loadLibrary("fastestdetv2");
    }

    public static native boolean init(AssetManager manager, String modelVariant);
    public static native Box[] detect(Bitmap bitmap, double threshold, double nms_threshold);
    public static native String bench(int iters);
}
