public class UpdateStatus { private ProgressDialog mPD ; private Context mCtx ; private Handler mHandler ; public UpdateStatus ( ) { } public UpdateStatus ( Context ctx , Handler handler , ProgressDialog pd ) { mCtx = ctx ; mPD = pd ; mHandler = handler ; } public void updateMessage ( int resId ) { if ( mCtx != null && mPD != null && mHandler != null ) { mHandler . post ( new UpdateMessage ( resId ) ) ; } } private class UpdateMessage implements Runnable { private int mResId ; public UpdateMessage ( int resId ) { mResId = resId ; } public void run ( ) { mPD . setMessage ( mCtx . getString ( mResId ) ) ; } } }