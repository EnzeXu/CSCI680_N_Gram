public class EntryEditActivityV4 extends EntryEditActivity { private ScrollView scroll; private LayoutInflater inflater; protected static void putParentId(Intent i, String parentKey, PwGroupV4 parent) { PwGroupId id = parent.getId(); PwGroupIdV4 id4 = (PwGroupIdV4) id; i.putExtra(parentKey, Types.UUIDtoBytes(id4.getId())); } @Override protected PwGroupId getParentGroupId(Intent i, String key) { byte[] buf = i.getByteArrayExtra(key); UUID id = Types.bytestoUUID(buf); return new PwGroupIdV4(id); } @Override protected void onCreate(Bundle savedInstanceState) { inflater = (LayoutInflater) this.getSystemService(Context.LAYOUT_INFLATER_SERVICE); super.onCreate(savedInstanceState); scroll = (ScrollView) findViewById(R.id.entry_scroll); ImageButton add = (ImageButton) findViewById(R.id.add_advanced); add.setVisibility(View.VISIBLE); add.setOnClickListener(new View.OnClickListener() { @Override public void onClick(View v) { LinearLayout container = (LinearLayout) findViewById(R.id.advanced_container); EntryEditSection ees = (EntryEditSection) inflater.inflate(R.layout.entry_edit_section, container, false); ees.setData("", new ProtectedString(false, "")); container.addView(ees); scroll.post(new Runnable() { @Override public void run() { scroll.fullScroll(ScrollView.FOCUS_DOWN); } }); } }); ImageButton iconPicker = (ImageButton) findViewById(R.id.icon_button); iconPicker.setVisibility(View.GONE); View divider = (View) findViewById(R.id.divider_title); RelativeLayout.LayoutParams lp_div = (RelativeLayout.LayoutParams) divider.getLayoutParams(); lp_div.addRule(RelativeLayout.BELOW, R.id.entry_title); View user_label = (View) findViewById(R.id.entry_user_name_label); RelativeLayout.LayoutParams lp = (RelativeLayout.LayoutParams) user_label.getLayoutParams(); lp.addRule(RelativeLayout.BELOW, R.id.divider_title); } @Override protected void fillData() { super.fillData(); PwEntryV4 entry = (PwEntryV4) mEntry; LinearLayout container = (LinearLayout) findViewById(R.id.advanced_container); if (entry.strings.size() > 0) { for (Entry<String, ProtectedString> pair : entry.strings.entrySet()) { String key = pair.getKey(); if (!PwEntryV4.IsStandardString(key)) { EntryEditSection ees = (EntryEditSection) inflater.inflate(R.layout.entry_edit_section, container, false); ees.setData(key, pair.getValue()); container.addView(ees); } } } } @SuppressWarnings("unchecked") @Override protected PwEntry populateNewEntry() { PwEntryV4 newEntry = (PwEntryV4) mEntry.clone(true); newEntry.history = (ArrayList<PwEntryV4>) newEntry.history.clone(); newEntry.createBackup((PwDatabaseV4)App.getDB().pm); newEntry = (PwEntryV4) super.populateNewEntry(newEntry); Map<String, ProtectedString> strings = newEntry.strings; Iterator<Entry<String, ProtectedString>> iter = strings.entrySet().iterator(); while (iter.hasNext()) { Entry<String, ProtectedString> pair = iter.next(); if (!PwEntryV4.IsStandardString(pair.getKey())) { iter.remove(); } } LinearLayout container = (LinearLayout) findViewById(R.id.advanced_container); for (int i = 0; i < container.getChildCount(); i++) { View view = container.getChildAt(i); TextView keyView = (TextView)view.findViewById(R.id.title); String key = keyView.getText().toString(); TextView valueView = (TextView)view.findViewById(R.id.value); String value = valueView.getText().toString(); CheckBox cb = (CheckBox)view.findViewById(R.id.protection); boolean protect = cb.isChecked(); strings.put(key, new ProtectedString(protect, value)); } return newEntry; } public void deleteAdvancedString(View view) { EntryEditSection section = (EntryEditSection) view.getParent(); LinearLayout container = (LinearLayout) findViewById(R.id.advanced_container); for (int i = 0; i < container.getChildCount(); i++) { EntryEditSection ees = (EntryEditSection) container.getChildAt(i); if (ees == section) { container.removeViewAt(i); container.invalidate(); break; } } } @Override protected boolean validateBeforeSaving() { if(!super.validateBeforeSaving()) { return false; } LinearLayout container = (LinearLayout) findViewById(R.id.advanced_container); for (int i = 0; i < container.getChildCount(); i++) { EntryEditSection ees = (EntryEditSection) container.getChildAt(i); TextView keyView = (TextView) ees.findViewById(R.id.title); CharSequence key = keyView.getText(); if (key == null || key.length() == 0) { Toast.makeText(this, R.string.error_string_key, Toast.LENGTH_LONG).show(); return false; } } return true; } }