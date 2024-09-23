public class CreateDB extends RunnableOnFinish { private final int DEFAULT_ENCRYPTION_ROUNDS = 300; private String mFilename; private boolean mDontSave; private String mDbName; private Context ctx; public CreateDB(Context ctx, String filename, String dbName, OnFinish finish, boolean dontSave) { super(finish); mFilename = filename; mDontSave = dontSave; mDbName = dbName; this.ctx = ctx; } @Override public void run() { Database db = new Database(); App.setDB(db); PwDatabase pm = PwDatabase.getNewDBInstance(mFilename); pm.initNew(mDbName); db.pm = pm; Uri.Builder b = new Uri.Builder(); db.mUri = UriUtil.parseDefaultFile(mFilename); db.setLoaded(); App.clearShutdown(); SaveDB save = new SaveDB(ctx, db, mFinish, mDontSave); mFinish = null; save.run(); } }