public class NativeLib { private static boolean isLoaded = false; private static boolean loadSuccess = false; public static boolean loaded() { return init(); } public static boolean init() { if ( ! isLoaded ) { try { System.loadLibrary("final-key"); System.loadLibrary("argon2"); } catch ( UnsatisfiedLinkError e) { return false; } isLoaded = true; loadSuccess = true; } return loadSuccess; } }