public class EntryActivity extends LockCloseHideActivity { public static final String KEY_ENTRY = "entry"; public static final String KEY_REFRESH_POS = "refresh_pos"; public static final int NOTIFY_USERNAME = 1; public static final int NOTIFY_PASSWORD = 2; public static void Launch(Activity act, PwEntry pw, int pos) { Intent i; if ( pw instanceof PwEntryV4 ) { i = new Intent(act, EntryActivityV4.class); } else { i = new Intent(act, EntryActivity.class); } i.putExtra(KEY_ENTRY, Types.UUIDtoBytes(pw.getUUID())); i.putExtra(KEY_REFRESH_POS, pos); act.startActivityForResult(i,0); } protected PwEntry mEntry; private Timer mTimer = new Timer(); private boolean mShowPassword; private int mPos; private NotificationManager mNM; private BroadcastReceiver mIntentReceiver; protected boolean readOnly = false; private DateFormat dateFormat; private DateFormat timeFormat; protected void setEntryView() { setContentView(R.layout.entry_view); } protected void setupEditButtons() { Button edit = (Button) findViewById(R.id.entry_edit); edit.setOnClickListener(new View.OnClickListener() { public void onClick(View v) { EntryEditActivity.Launch(EntryActivity.this, mEntry); } }); if (readOnly) { edit.setVisibility(View.GONE); View divider = findViewById(R.id.entry_divider2); divider.setVisibility(View.GONE); } } @Override protected void onCreate(Bundle savedInstanceState) { SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this); mShowPassword = ! prefs.getBoolean(getString(R.string.maskpass_key), getResources().getBoolean(R.bool.maskpass_default)); super.onCreate(savedInstanceState); setEntryView(); Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar); setSupportActionBar(toolbar); Context appCtx = getApplicationContext(); dateFormat = android.text.format.DateFormat.getDateFormat(appCtx); timeFormat = android.text.format.DateFormat.getTimeFormat(appCtx); Database db = App.getDB(); if ( ! db.Loaded() ) { finish(); return; } readOnly = db.readOnly; setResult(KeePass.EXIT_NORMAL); Intent i = getIntent(); UUID uuid = Types.bytestoUUID(i.getByteArrayExtra(KEY_ENTRY)); mPos = i.getIntExtra(KEY_REFRESH_POS, -1); assert(uuid != null); mEntry = db.pm.entries.get(uuid); if (mEntry == null) { Toast.makeText(this, R.string.entry_not_found, Toast.LENGTH_LONG).show(); finish(); return; } this.invalidateOptionsMenu(); mEntry.touch(false, false); fillData(false); setupEditButtons(); NotificationUtil.createChannels(getApplicationContext()); mNM = (NotificationManager) getSystemService(NOTIFICATION_SERVICE); if ( mEntry.getPassword().length() > 0 ) { Notification password = getNotification(Intents.COPY_PASSWORD, R.string.copy_password); mNM.notify(NOTIFY_PASSWORD, password); } if ( mEntry.getUsername().length() > 0 ) { Notification username = getNotification(Intents.COPY_USERNAME, R.string.copy_username); mNM.notify(NOTIFY_USERNAME, username); } mIntentReceiver = new BroadcastReceiver() { @Override public void onReceive(Context context, Intent intent) { String action = intent.getAction(); if ( action.equals(Intents.COPY_USERNAME) ) { String username = mEntry.getUsername(); if ( username.length() > 0 ) { timeoutCopyToClipboard(getString(R.string.hint_username), username); } } else if ( action.equals(Intents.COPY_PASSWORD) ) { String password = new String(mEntry.getPassword()); if ( password.length() > 0 ) { timeoutCopyToClipboard(getString(R.string.hint_login_pass), new String(mEntry.getPassword()), true); } } } }; IntentFilter filter = new IntentFilter(); filter.addAction(Intents.COPY_USERNAME); filter.addAction(Intents.COPY_PASSWORD); registerReceiver(mIntentReceiver, filter); } @Override protected void onDestroy() { if ( mIntentReceiver != null ) { unregisterReceiver(mIntentReceiver); } if ( mNM != null ) { try { mNM.cancelAll(); } catch (SecurityException e) { } } super.onDestroy(); } private Notification getNotification(String intentText, int descResId) { String desc = getString(descResId); Intent intent = new Intent(intentText); int flags = PendingIntent.FLAG_CANCEL_CURRENT; if (Build.VERSION.SDK_INT >= 23) { flags |= PendingIntent.FLAG_IMMUTABLE; } PendingIntent pending = PendingIntent.getBroadcast(this, 0, intent, flags); NotificationCompat.Builder builder = new NotificationCompat.Builder(this, NotificationUtil.COPY_CHANNEL_ID); Notification notify = builder.setContentIntent(pending).setContentText(desc).setContentTitle(getString(R.string.app_name)) .setSmallIcon(R.drawable.notify).setTicker(desc).setWhen(System.currentTimeMillis()).build(); return notify; } private String getDateTime(Date dt) { return dateFormat.format(dt) + " " + timeFormat.format(dt); } protected void fillData(boolean trimList) { ImageView iv = (ImageView) findViewById(R.id.entry_icon); Database db = App.getDB(); db.drawFactory.assignDrawableTo(iv, getResources(), mEntry.getIcon()); PwDatabase pm = db.pm; populateText(R.id.entry_title, mEntry.getTitle(true, pm)); populateText(R.id.entry_user_name, mEntry.getUsername(true, pm)); populateText(R.id.entry_url, mEntry.getUrl(true, pm)); populateText(R.id.entry_password, mEntry.getPassword(true, pm)); setPasswordStyle(); populateText(R.id.entry_created, getDateTime(mEntry.getCreationTime())); populateText(R.id.entry_modified, getDateTime(mEntry.getLastModificationTime())); populateText(R.id.entry_accessed, getDateTime(mEntry.getLastAccessTime())); Date expires = mEntry.getExpiryTime(); if ( mEntry.expires() ) { populateText(R.id.entry_expires, getDateTime(expires)); } else { populateText(R.id.entry_expires, R.string.never); } populateText(R.id.entry_comment, mEntry.getNotes(true, pm)); } private void populateText(int viewId, int resId) { TextView tv = (TextView) findViewById(viewId); tv.setText(resId); } private void populateText(int viewId, String text) { TextView tv = (TextView) findViewById(viewId); tv.setText(text); } @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) { super.onActivityResult(requestCode, resultCode, data); if ( resultCode == KeePass.EXIT_REFRESH || resultCode == KeePass.EXIT_REFRESH_TITLE ) { fillData(true); if ( resultCode == KeePass.EXIT_REFRESH_TITLE ) { Intent ret = new Intent(); ret.putExtra(KEY_REFRESH_POS, mPos); setResult(KeePass.EXIT_REFRESH, ret); } } } @Override public boolean onCreateOptionsMenu(Menu menu) { super.onCreateOptionsMenu(menu); MenuInflater inflater = getMenuInflater(); inflater.inflate(R.menu.entry, menu); MenuItem togglePassword = menu.findItem(R.id.menu_toggle_pass); if ( mShowPassword ) { togglePassword.setTitle(R.string.menu_hide_password); } else { togglePassword.setTitle(R.string.menu_showpass); } MenuItem gotoUrl = menu.findItem(R.id.menu_goto_url); MenuItem copyUser = menu.findItem(R.id.menu_copy_user); MenuItem copyPass = menu.findItem(R.id.menu_copy_pass); if (mEntry == null) { gotoUrl.setVisible(false); copyUser.setVisible(false); copyPass.setVisible(false); } else { String url = mEntry.getUrl(); if (EmptyUtils.isNullOrEmpty(url)) { gotoUrl.setVisible(false); } if ( mEntry.getUsername().length() == 0 ) { copyUser.setVisible(false); } if ( mEntry.getPassword().length() == 0 ) { copyPass.setVisible(false); } } return true; } private void setPasswordStyle() { TextView password = (TextView) findViewById(R.id.entry_password); if ( mShowPassword ) { password.setTransformationMethod(null); } else { password.setTransformationMethod(PasswordTransformationMethod.getInstance()); } } @Override public boolean onOptionsItemSelected(MenuItem item) { switch ( item.getItemId() ) { case R.id.menu_donate: try { Util.gotoUrl(this, R.string.donate_url); } catch (ActivityNotFoundException e) { Toast.makeText(this, R.string.error_failed_to_launch_link, Toast.LENGTH_LONG).show(); return false; } return true; case R.id.menu_toggle_pass: if ( mShowPassword ) { item.setTitle(R.string.menu_showpass); mShowPassword = false; } else { item.setTitle(R.string.menu_hide_password); mShowPassword = true; } setPasswordStyle(); return true; case R.id.menu_goto_url: String url; url = mEntry.getUrl(); if ( ! url.contains(": url = "http: } try { Util.gotoUrl(this, url); } catch (ActivityNotFoundException e) { Toast.makeText(this, R.string.no_url_handler, Toast.LENGTH_LONG).show(); } return true; case R.id.menu_copy_user: timeoutCopyToClipboard(getString(R.string.hint_username), mEntry.getUsername(true, App.getDB().pm)); return true; case R.id.menu_copy_pass: timeoutCopyToClipboard(getString(R.string.hint_login_pass), new String(mEntry.getPassword(true, App.getDB().pm)), true); return true; case R.id.menu_lock: App.setShutdown(); setResult(KeePass.EXIT_LOCK); finish(); return true; } return super.onOptionsItemSelected(item); }