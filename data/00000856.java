public class Interaction { public static boolean isIntentAvailable(Context context, String action) { final PackageManager packageManager = context.getPackageManager(); final Intent intent = new Intent(action); List<ResolveInfo> list = packageManager.queryIntentActivities(intent, PackageManager.MATCH_DEFAULT_ONLY); return list.size() > 0; } }