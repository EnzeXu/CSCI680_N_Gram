public class StorageAF { public static String ACTION_OPEN_DOCUMENT; static { try { Field openDocument = Intent.class.getField("ACTION_OPEN_DOCUMENT"); ACTION_OPEN_DOCUMENT = (String) openDocument.get(null); } catch (Exception e) { ACTION_OPEN_DOCUMENT = "android.intent.action.OPEN_DOCUMENT"; } } public static boolean supportsStorageFramework() { return Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT; } public static boolean useStorageFramework(Context ctx) { if (!supportsStorageFramework()) { return false; } SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(ctx); return prefs.getBoolean(ctx.getString(R.string.saf_key), ctx.getResources().getBoolean(R.bool.saf_default)); } }