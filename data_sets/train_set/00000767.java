public class AddGroup extends RunnableOnFinish { protected Database mDb; private String mName; private int mIconID; private PwGroup mGroup; private PwGroup mParent; private Context ctx; protected boolean mDontSave; public static AddGroup getInstance(Context ctx, Database db, String name, int iconid, PwGroup parent, OnFinish finish, boolean dontSave) { return new AddGroup(ctx, db, name, iconid, parent, finish, dontSave); } private AddGroup(Context ctx, Database db, String name, int iconid, PwGroup parent, OnFinish finish, boolean dontSave) { super(finish); mDb = db; mName = name; mIconID = iconid; mParent = parent; mDontSave = dontSave; this.ctx = ctx; mFinish = new AfterAdd(mFinish); } @Override public void run() { PwDatabase pm = mDb.pm; mGroup = pm.createGroup(); mGroup.initNewGroup(mName, pm.newGroupId()); mGroup.icon = mDb.pm.iconFactory.getIcon(mIconID); pm.addGroupTo(mGroup, mParent); SaveDB save = new SaveDB(ctx, mDb, mFinish, mDontSave); save.run(); } private class AfterAdd extends OnFinish { public AfterAdd(OnFinish finish) { super(finish); } @Override public void run() { PwDatabase pm = mDb.pm; if ( mSuccess ) { mDb.dirty.add(mParent); } else { pm.removeGroupFrom(mGroup, mParent); } super.run(); } } }