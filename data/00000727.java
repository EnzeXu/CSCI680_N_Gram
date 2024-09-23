public class PwGroupListAdapter extends BaseAdapter { private GroupBaseActivity mAct; private PwGroup mGroup; private List<PwGroup> groupsForViewing; private List<PwEntry> entriesForViewing; private Comparator<PwEntry> entryComp = new PwEntry.EntryNameComparator(); private Comparator<PwGroup> groupComp = new PwGroup.GroupNameComparator(); private SharedPreferences prefs; public PwGroupListAdapter(GroupBaseActivity act, PwGroup group) { mAct = act; mGroup = group; prefs = PreferenceManager.getDefaultSharedPreferences(act); filterAndSort(); } @Override public void notifyDataSetChanged() { super.notifyDataSetChanged(); filterAndSort(); } @Override public void notifyDataSetInvalidated() { super.notifyDataSetInvalidated(); filterAndSort(); } private void filterAndSort() { entriesForViewing = new ArrayList<PwEntry>(); for (int i = 0; i < mGroup.childEntries.size(); i++) { PwEntry entry = mGroup.childEntries.get(i); if ( ! entry.isMetaStream() ) { entriesForViewing.add(entry); } } boolean sortLists = prefs.getBoolean(mAct.getString(R.string.sort_key), mAct.getResources().getBoolean(R.bool.sort_default)); if ( sortLists ) { groupsForViewing = new ArrayList<PwGroup>(mGroup.childGroups); Collections.sort(entriesForViewing, entryComp); Collections.sort(groupsForViewing, groupComp); } else { groupsForViewing = mGroup.childGroups; } } public int getCount() { return groupsForViewing.size() + entriesForViewing.size(); } public Object getItem(int position) { return position; } public long getItemId(int position) { return position; } public View getView(int position, View convertView, ViewGroup parent) { int size = groupsForViewing.size(); if ( position < size ) { return createGroupView(position, convertView); } else { return createEntryView(position - size, convertView); } } private View createGroupView(int position, View convertView) { PwGroup group = groupsForViewing.get(position); PwGroupView gv; if (convertView == null || !(convertView instanceof PwGroupView)) { gv = PwGroupView.getInstance(mAct, group); } else { gv = (PwGroupView) convertView; gv.convertView(group); } return gv; } private PwEntryView createEntryView(int position, View convertView) { PwEntry entry = entriesForViewing.get(position); PwEntryView ev; if (convertView == null || !(convertView instanceof PwEntryView)) { ev = PwEntryView.getInstance(mAct, entry, position); } else { ev = (PwEntryView) convertView; ev.convertView(entry, position); } return ev; } }