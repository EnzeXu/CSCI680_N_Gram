public class EditHostActivity extends AppCompatActivity implements HostEditorFragment.Listener { private static final String EXTRA_EXISTING_HOST_ID = "org.connectbot.existing_host_id"; private static final long NO_HOST_ID = -1; private static final int ENABLED_ALPHA = 255; private static final int DISABLED_ALPHA = 130; private HostDatabase mHostDb; private PubkeyDatabase mPubkeyDb; private ServiceConnection mTerminalConnection; private HostBean mHost; private TerminalBridge mBridge; private boolean mIsCreating; private MenuItem mSaveHostButton; public static Intent createIntentForExistingHost(Context context, long existingHostId) { Intent i = new Intent(context, EditHostActivity.class); i.putExtra(EXTRA_EXISTING_HOST_ID, existingHostId); return i; } public static Intent createIntentForNewHost(Context context) { return createIntentForExistingHost(context, NO_HOST_ID); } @Override protected void onCreate(Bundle savedInstanceState) { super.onCreate(savedInstanceState); mHostDb = HostDatabase.get(this); mPubkeyDb = PubkeyDatabase.get(this); mTerminalConnection = new ServiceConnection() { @Override public void onServiceConnected(ComponentName className, IBinder service) { TerminalManager bound = ((TerminalManager.TerminalBinder) service).getService(); mBridge = bound.getConnectedBridge(mHost); } @Override public void onServiceDisconnected(ComponentName name) { mBridge = null; } }; long hostId = getIntent().getLongExtra(EXTRA_EXISTING_HOST_ID, NO_HOST_ID); mIsCreating = hostId == NO_HOST_ID; mHost = mIsCreating ? null : mHostDb.findHostById(hostId); ArrayList<String> pubkeyNames = new ArrayList<>(); ArrayList<String> pubkeyValues = new ArrayList<>(); TypedArray defaultPubkeyNames = getResources().obtainTypedArray(R.array.list_pubkeyids); for (int i = 0; i < defaultPubkeyNames.length(); i++) { pubkeyNames.add(defaultPubkeyNames.getString(i)); } TypedArray defaultPubkeyValues = getResources().obtainTypedArray(R.array.list_pubkeyids_value); for (int i = 0; i < defaultPubkeyValues.length(); i++) { pubkeyValues.add(defaultPubkeyValues.getString(i)); } for (CharSequence cs : mPubkeyDb.allValues(PubkeyDatabase.FIELD_PUBKEY_NICKNAME)) { pubkeyNames.add(cs.toString()); } for (CharSequence cs : mPubkeyDb.allValues("_id")) { pubkeyValues.add(cs.toString()); } setContentView(R.layout.activity_edit_host); FragmentManager fm = getSupportFragmentManager(); HostEditorFragment fragment = (HostEditorFragment) fm.findFragmentById(R.id.fragment_container); if (fragment == null) { fragment = HostEditorFragment.newInstance(mHost, pubkeyNames, pubkeyValues); getSupportFragmentManager().beginTransaction() .add(R.id.fragment_container, fragment).commit(); } defaultPubkeyNames.recycle(); defaultPubkeyValues.recycle(); } @Override public boolean onCreateOptionsMenu(Menu menu) { MenuInflater inflater = getMenuInflater(); inflater.inflate( mIsCreating ? R.menu.edit_host_activity_add_menu : R.menu.edit_host_activity_edit_menu, menu); mSaveHostButton = menu.getItem(0); setAddSaveButtonEnabled(!mIsCreating); return super.onCreateOptionsMenu(menu); } @Override public boolean onOptionsItemSelected(MenuItem item) { int itemId = item.getItemId(); if (itemId == R.id.save || itemId == android.R.id.home) { attemptSaveAndExit(); return true; } return super.onOptionsItemSelected(item); } @Override public void onStart() { super.onStart(); bindService(new Intent( this, TerminalManager.class), mTerminalConnection, Context.BIND_AUTO_CREATE); final HostEditorFragment fragment = (HostEditorFragment) getSupportFragmentManager(). findFragmentById(R.id.fragment_container); if (CharsetHolder.isInitialized()) { fragment.setCharsetData(CharsetHolder.getCharsetData()); } else { AsyncTask<Void, Void, Void> charsetTask = new AsyncTask<Void, Void, Void>() { @Override protected Void doInBackground(Void... unused) { CharsetHolder.initialize(); return null; } @Override protected void onPostExecute(Void unused) { fragment.setCharsetData(CharsetHolder.getCharsetData()); } }; charsetTask.execute(); } } @Override public void onStop() { super.onStop(); unbindService(mTerminalConnection); } @Override public void onValidHostConfigured(HostBean host) { mHost = host; if (mSaveHostButton != null) setAddSaveButtonEnabled(true); } @Override public void onHostInvalidated() { mHost = null; if (mSaveHostButton != null) setAddSaveButtonEnabled(false); } @Override public void onBackPressed() { attemptSaveAndExit(); } private void attemptSaveAndExit() { if (mHost == null) { showDiscardDialog(); return; } mHostDb.saveHost(mHost); if (mBridge != null) { mBridge.setCharset(mHost.getEncoding()); } finish(); } private void showDiscardDialog() { androidx.appcompat.app.AlertDialog.Builder builder = new androidx.appcompat.app.AlertDialog.Builder(this, R.style.AlertDialogTheme); builder.setMessage(R.string.discard_host_changes_message) .setPositiveButton(R.string.discard_host_button, new DialogInterface.OnClickListener() { @Override public void onClick(DialogInterface dialog, int which) { finish(); } }) .setNegativeButton(R.string.discard_host_cancel_button, new DialogInterface.OnClickListener() { @Override public void onClick(DialogInterface dialog, int which) { } }); builder.show(); } private void setAddSaveButtonEnabled(boolean enabled) { mSaveHostButton.setEnabled(enabled); mSaveHostButton.getIcon().setAlpha(enabled ? ENABLED_ALPHA : DISABLED_ALPHA); } private static class CharsetHolder { private static boolean mInitialized = false; private static Map<String, String> mData; public static Map<String, String> getCharsetData() { if (mData == null) initialize(); return mData; } private synchronized static void initialize() { if (mInitialized) return; mData = new HashMap<>(); for (Map.Entry<String, Charset> entry : Charset.availableCharsets().entrySet()) { Charset c = entry.getValue(); if (c.canEncode() && c.isRegistered()) { String key = entry.getKey(); if (key.startsWith("cp")) { mData.put("CP437", "CP437"); } mData.put(c.displayName(), entry.getKey()); } } mInitialized = true; } public static boolean isInitialized() { return mInitialized; } } }