#include <jni.h>
#include <string>
#include <android/asset_manager_jni.h>
#include "FastestDetV2.h"

namespace {

void native_init(JNIEnv *env, jclass, jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        return;
    }

    delete FastestDetV2::detector;
    FastestDetV2::detector = new FastestDetV2(mgr, "fastestdetv2.param", "fastestdetv2.bin");
}

jobjectArray native_detect(JNIEnv *env, jclass, jobject image, jdouble threshold, jdouble nms_threshold) {
    auto box_cls = env->FindClass("org/eu/pnxlr/git/pnxlr/fastestdetv2/Box");
    jobjectArray ret = env->NewObjectArray(0, box_cls, nullptr);
    if (FastestDetV2::detector == nullptr || !FastestDetV2::detector->isLoaded() || image == nullptr) {
        return ret;
    }

    auto result = FastestDetV2::detector->detect(env, image, threshold, nms_threshold);

    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.x2, box.y2, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}

jstring native_bench(JNIEnv *env, jclass, jint iters) {
    if (FastestDetV2::detector == nullptr) {
        return env->NewStringUTF("FastestDetV2 is not initialized.");
    }
    if (!FastestDetV2::detector->isLoaded()) {
        return env->NewStringUTF("FastestDetV2 model load failed.");
    }
    std::string result = FastestDetV2::detector->bench(iters);
    return env->NewStringUTF(result.c_str());
}

} // namespace

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK || env == nullptr) {
        return -1;
    }

    jclass clazz = env->FindClass("org/eu/pnxlr/git/pnxlr/fastestdetv2/FastestDetV2");
    if (clazz == nullptr) {
        return -1;
    }

    static const JNINativeMethod methods[] = {
            {"init", "(Landroid/content/res/AssetManager;)V", reinterpret_cast<void *>(native_init)},
            {"detect", "(Landroid/graphics/Bitmap;DD)[Lorg/eu/pnxlr/git/pnxlr/fastestdetv2/Box;", reinterpret_cast<void *>(native_detect)},
            {"bench", "(I)Ljava/lang/String;", reinterpret_cast<void *>(native_bench)},
    };
    if (env->RegisterNatives(clazz, methods, sizeof(methods) / sizeof(methods[0])) != JNI_OK) {
        return -1;
    }

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
}
