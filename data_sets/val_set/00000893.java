public class KeePass extends Activity { public static final int EXIT_NORMAL = 0; public static final int EXIT_LOCK = 1; public static final int EXIT_REFRESH = 2; public static final int EXIT_REFRESH_TITLE = 3; @Override protected void onCreate(Bundle savedInstanceState) { super.onCreate(savedInstanceState); } @Override protected void onStart() { super.onStart(); startFileSelect(); } private void startFileSelect() { Intent intent = new Intent(this, FileSelectActivity.class); startActivityForResult(intent, 0); } @Override protected void onDestroy() { super.onDestroy(); } @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) { super.onActivityResult(requestCode, resultCode, data); if (resultCode == EXIT_NORMAL) { finish(); } } }