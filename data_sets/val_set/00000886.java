public class PwGroupViewV3 extends PwGroupView { private static final int MENU_DELETE = MENU_OPEN + 1; protected PwGroupViewV3(GroupBaseActivity act, PwGroup pw) { super(act, pw); } @Override public void onCreateMenu(ContextMenu menu, ContextMenuInfo menuInfo) { super.onCreateMenu(menu, menuInfo); if (!readOnly) { menu.add(0, MENU_DELETE, 0, R.string.menu_delete); } } @Override public boolean onContextItemSelected(MenuItem item) { if ( ! super.onContextItemSelected(item) ) { switch ( item.getItemId() ) { case MENU_DELETE: Handler handler = new Handler(); DeleteGroup task = new DeleteGroup(App.getDB(), mPw, mAct, mAct.new AfterDeleteGroup(handler)); ProgressTask pt = new ProgressTask(mAct, task, R.string.saving_database); pt.run(); return true; } } return false; } }