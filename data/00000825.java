public class ContentResolverCompat { public static boolean available; private static Class contentResolver; private static Method takePersistableUriPermission; static { try { contentResolver = ContentResolver.class; takePersistableUriPermission = contentResolver.getMethod("takePersistableUriPermission", new Class[]{Uri.class, int.class}); available = true; } catch (Exception e) { available = false; } } public static void takePersistableUriPermission(ContentResolver resolver, Uri uri, int modeFlags) { if (available) { try { takePersistableUriPermission.invoke(resolver, new Object[]{uri, modeFlags}); } catch (Exception e) { } } } }