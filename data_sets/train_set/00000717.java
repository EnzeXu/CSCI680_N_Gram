public class GroupActivityV4 extends GroupActivity { @Override protected PwGroupId retrieveGroupId(Intent i) { String uuid = i.getStringExtra(KEY_ENTRY); if ( uuid == null || uuid.length() == 0 ) { return null; } return new PwGroupIdV4(UUID.fromString(uuid)); } @Override protected void setupButtons() { super.setupButtons(); addEntryEnabled = !readOnly; } }