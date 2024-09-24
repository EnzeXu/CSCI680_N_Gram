public class SaveDB extends RunnableOnFinish { private Database mDb; private boolean mDontSave; private Context mCtx; public SaveDB(Context ctx, Database db, OnFinish finish, boolean dontSave) { super(finish); mDb = db; mDontSave = dontSave; mCtx = ctx; } public SaveDB(Context ctx, Database db, OnFinish finish) { super(finish); mDb = db; mDontSave = false; mCtx = ctx; } @Override public void run() { if ( ! mDontSave ) { try { mDb.SaveData(mCtx); } catch (IOException e) { finish(false, e.getMessage()); return; } catch (FileUriException e) { if (Android11WarningFragment.showAndroid11WarningOnThisVersion()) { finish(false, new Android11WarningFragment(R.string.Android11SaveFailed)); } else { finish(false, e.getMessage()); } return; } catch (PwDbOutputException e) { throw new RuntimeException(e); } } finish(true); } }