public class LockingClosePreferenceActivity extends LockingPreferenceActivity { @ Override protected void onResume ( ) { super . onResume ( ) ; TimeoutHelper . checkShutdown ( this ) ; } }