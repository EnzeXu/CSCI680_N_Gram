public class FinalKeyFactory { public static FinalKey createFinalKey() { return createFinalKey(false); } public static FinalKey createFinalKey(boolean androidOverride) { if ( !CipherFactory.deviceBlacklisted() && !androidOverride && NativeFinalKey.availble() ) { return new NativeFinalKey(); } else { return new AndroidFinalKey(); } } }