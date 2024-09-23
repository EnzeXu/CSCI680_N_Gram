public class TerminalView extends FrameLayout implements FontSizeChangedListener { private final Context context ; public final TerminalBridge bridge ; private final TerminalTextViewOverlay terminalTextViewOverlay ; public final TerminalViewPager viewPager ; private final GestureDetector gestureDetector ; private final SharedPreferences prefs ; private int lastTouchedRow , lastTouchedCol ; private final ClipboardManager clipboard ; private final Paint paint ; private final Paint cursorPaint ; private final Paint cursorStrokePaint ; private final Paint cursorInversionPaint ; private final Paint cursorMetaInversionPaint ; private final Path ctrlCursor ; private final Path altCursor ; private final Path shiftCursor ; private final RectF tempSrc ; private final RectF tempDst ; private final Matrix scaleMatrix ; private static final Matrix . ScaleToFit scaleType = Matrix . ScaleToFit . FILL ; private Toast notification = null ; private String lastNotification = null ; private volatile boolean notifications = true ; private boolean mAccessibilityInitialized = false ; private boolean mAccessibilityActive = true ; private final Object [ ] mAccessibilityLock = new Object [ 0 ] ; private final StringBuffer mAccessibilityBuffer ; private Pattern mControlCodes = null ; private Matcher mCodeMatcher = null ; private AccessibilityEventSender mEventSender = null ; private final char [ ] singleDeadKey = new char [ 1 ] ; private static final String BACKSPACE_CODE = "\\x08\\x1b\\ [ K" ; private static final String CONTROL_CODE_PATTERN = "\\x1b\\ [ K [ ^m ] + [ m| : ] " ; private static final int ACCESSIBILITY_EVENT_THRESHOLD = 1000 ; private static final String SCREENREADER_INTENT_ACTION = "android . accessibilityservice . AccessibilityService" ; private static final String SCREENREADER_INTENT_CATEGORY = "android . accessibilityservice . category . FEEDBACK_SPOKEN" ; public TerminalView ( Context context , TerminalBridge bridge , TerminalViewPager pager ) { super ( context ) ; setWillNotDraw ( false ) ; this . context = context ; this . bridge = bridge ; this . viewPager = pager ; mAccessibilityBuffer = new StringBuffer ( ) ; setLayoutParams ( new LayoutParams ( LayoutParams . FILL_PARENT , LayoutParams . FILL_PARENT ) ) ; setFocusable ( true ) ; setFocusableInTouchMode ( true ) ; setLayerTypeToSoftware ( ) ; paint = new Paint ( ) ; cursorPaint = new Paint ( ) ; cursorPaint . setColor ( bridge . color [ bridge . defaultFg ] ) ; cursorPaint . setAntiAlias ( true ) ; cursorInversionPaint = new Paint ( ) ; cursorInversionPaint . setColorFilter ( new ColorMatrixColorFilter ( new ColorMatrix ( new float [ ] { -1 , 0 , 0 , 0 , 255 , 0 , -1 , 0 , 0 , 255 , 0 , 0 , -1 , 0 , 255 , 0 , 0 , 0 , 1 , 0 } ) ) ) ; cursorInversionPaint . setAntiAlias ( true ) ; cursorMetaInversionPaint = new Paint ( ) ; cursorMetaInversionPaint . setColorFilter ( new ColorMatrixColorFilter ( new ColorMatrix ( new float [ ] { -1f , 0 , 0 , 0 , 255 , 0 , -1f , 0 , 0 , 255 , 0 , 0 , -1f , 0 , 255 , 0 , 0 , 0 , 0 . 5f , 0 } ) ) ) ; cursorMetaInversionPaint . setAntiAlias ( true ) ; cursorStrokePaint = new Paint ( cursorInversionPaint ) ; cursorStrokePaint . setStrokeWidth ( 0 . 1f ) ; cursorStrokePaint . setStyle ( Paint . Style . STROKE ) ; shiftCursor = new Path ( ) ; shiftCursor . lineTo ( 0 . 5f , 0 . 33f ) ; shiftCursor . lineTo ( 1 . 0f , 0 . 0f ) ; altCursor = new Path ( ) ; altCursor . moveTo ( 0 . 0f , 1 . 0f ) ; altCursor . lineTo ( 0 . 5f , 0 . 66f ) ; altCursor . lineTo ( 1 . 0f , 1 . 0f ) ; ctrlCursor = new Path ( ) ; ctrlCursor . moveTo ( 0 . 0f , 0 . 25f ) ; ctrlCursor . lineTo ( 1 . 0f , 0 . 5f ) ; ctrlCursor . lineTo ( 0 . 0f , 0 . 75f ) ; tempSrc = new RectF ( ) ; tempSrc . set ( 0 . 0f , 0 . 0f , 1 . 0f , 1 . 0f ) ; tempDst = new RectF ( ) ; scaleMatrix = new Matrix ( ) ; setOnKeyListener ( bridge . getKeyHandler ( ) ) ; terminalTextViewOverlay = new TerminalTextViewOverlay ( context , this ) ; terminalTextViewOverlay . setLayoutParams ( new RelativeLayout . LayoutParams ( LayoutParams . MATCH_PARENT , LayoutParams . MATCH_PARENT ) ) ; addView ( terminalTextViewOverlay , 0 ) ; terminalTextViewOverlay . setOnKeyListener ( bridge . getKeyHandler ( ) ) ; clipboard = ( ClipboardManager ) context . getSystemService ( Context . CLIPBOARD_SERVICE ) ; prefs = PreferenceManager . getDefaultSharedPreferences ( context ) ; bridge . addFontSizeChangedListener ( this ) ; bridge . parentChanged ( this ) ; onFontSizeChanged ( 0 ) ; gestureDetector = new GestureDetector ( context , new GestureDetector . SimpleOnGestureListener ( ) { private final TerminalBridge bridge = TerminalView . this . bridge ; private float totalY = 0 ; @ Override public boolean onScroll ( MotionEvent e1 , MotionEvent e2 , float distanceX , float distanceY ) { int touchSlop = ViewConfiguration . get ( TerminalView . this . context ) . getScaledTouchSlop ( ) ; if ( Math . abs ( e1 . getX ( ) - e2 . getX ( ) ) < touchSlop * 4 ) { totalY += distanceY ; final int moved = ( int ) ( totalY / bridge . charHeight ) ; boolean pgUpDnGestureEnabled = prefs . getBoolean ( PreferenceConstants . PG_UPDN_GESTURE , false ) ; if ( pgUpDnGestureEnabled && e2 . getX ( ) < = getWidth ( ) / 3 ) { if ( moved > 5 ) { ( ( vt320 ) bridge . buffer ) . keyPressed ( vt320 . KEY_PAGE_DOWN , ' ' , 0 ) ; bridge . tryKeyVibrate ( ) ; totalY = 0 ; } else if ( moved < -5 ) { ( ( vt320 ) bridge . buffer ) . keyPressed ( vt320 . KEY_PAGE_UP , ' ' , 0 ) ; bridge . tryKeyVibrate ( ) ; totalY = 0 ; } return true ; } else if ( moved != 0 ) { int base = bridge . buffer . getWindowBase ( ) ; bridge . buffer . setWindowBase ( base + moved ) ; totalY = 0 ; return false ; } } return false ; } @ Override public boolean onSingleTapConfirmed ( MotionEvent e ) { viewPager . performClick ( ) ; return super . onSingleTapConfirmed ( e ) ; } } ) ; new AccessibilityStateTester ( ) . execute ( ( Void ) null ) ; } private void setLayerTypeToSoftware ( ) { setLayerType ( View . LAYER_TYPE_SOFTWARE , null ) ; } public void copyCurrentSelectionToClipboard ( ) { if ( terminalTextViewOverlay != null ) { terminalTextViewOverlay . copyCurrentSelectionToClipboard ( ) ; } } @ Override public boolean onTouchEvent ( MotionEvent event ) { if ( gestureDetector != null && gestureDetector . onTouchEvent ( event ) ) { return true ; } if ( terminalTextViewOverlay == null ) { if ( bridge . isSelectingForCopy ( ) ) { SelectionArea area = bridge . getSelectionArea ( ) ; int row = ( int ) Math . floor ( event . getY ( ) / bridge . charHeight ) ; int col = ( int ) Math . floor ( event . getX ( ) / bridge . charWidth ) ; switch ( event . getAction ( ) ) { case MotionEvent . ACTION_DOWN : viewPager . setPagingEnabled ( false ) ; if ( area . isSelectingOrigin ( ) ) { area . setRow ( row ) ; area . setColumn ( col ) ; lastTouchedRow = row ; lastTouchedCol = col ; bridge . redraw ( ) ; } return true ; case MotionEvent . ACTION_MOVE : if ( row == lastTouchedRow && col == lastTouchedCol ) return true ; area . finishSelectingOrigin ( ) ; area . setRow ( row ) ; area . setColumn ( col ) ; lastTouchedRow = row ; lastTouchedCol = col ; bridge . redraw ( ) ; return true ; case MotionEvent . ACTION_UP : if ( area . getLeft ( ) == area . getRight ( ) && area . getTop ( ) == area . getBottom ( ) ) { return true ; } String copiedText = area . copyFrom ( bridge . buffer ) ; clipboard . setText ( copiedText ) ; Toast . makeText ( context , context . getResources ( ) . getQuantityString ( R . plurals . console_copy_done , copiedText . length ( ) , copiedText . length ( ) ) , Toast . LENGTH_LONG ) . show ( ) ; case MotionEvent . ACTION_CANCEL : area . reset ( ) ; bridge . setSelectingForCopy ( false ) ; bridge . redraw ( ) ; viewPager . setPagingEnabled ( true ) ; return true ; } } return true ; } return super . onTouchEvent ( event ) ; } @ Override protected void onSizeChanged ( int w , int h , int oldw , int oldh ) { super . onSizeChanged ( w , h , oldw , oldh ) ; bridge . parentChanged ( this ) ; scaleCursors ( ) ; } @ Override public void onFontSizeChanged ( final float unusedSizeDp ) { scaleCursors ( ) ; ( ( Activity ) context ) . runOnUiThread ( new Runnable ( ) { @ Override public void run ( ) { if ( terminalTextViewOverlay != null ) { terminalTextViewOverlay . setTextSize ( TypedValue . COMPLEX_UNIT_PX , bridge . getTextSizePx ( ) ) ; float lineSpacingMultiplier = ( float ) bridge . charHeight / terminalTextViewOverlay . getPaint ( ) . getFontMetricsInt ( null ) ; terminalTextViewOverlay . setLineSpacing ( 0 . 0f , lineSpacingMultiplier ) ; } } } ) ; } private void scaleCursors ( ) { tempDst . set ( 0 . 0f , 0 . 0f , bridge . charWidth , bridge . charHeight ) ; scaleMatrix . setRectToRect ( tempSrc , tempDst , scaleType ) ; } @ Override public void onDraw ( Canvas canvas ) { if ( bridge . bitmap != null ) { bridge . onDraw ( ) ; canvas . drawBitmap ( bridge . bitmap , 0 , 0 , paint ) ; if ( bridge . buffer . isCursorVisible ( ) ) { int cursorColumn = bridge . buffer . getCursorColumn ( ) ; final int cursorRow = bridge . buffer . getCursorRow ( ) ; final int columns = bridge . buffer . getColumns ( ) ; if ( cursorColumn == columns ) cursorColumn = columns - 1 ; if ( cursorColumn < 0 || cursorRow < 0 ) return ; long currentAttribute = bridge . buffer . getAttributes ( cursorColumn , cursorRow ) ; boolean onWideCharacter = ( currentAttribute & VDUBuffer . FULLWIDTH ) != 0 ; int x = cursorColumn * bridge . charWidth ; int y = ( bridge . buffer . getCursorRow ( ) + bridge . buffer . screenBase - bridge . buffer . windowBase ) * bridge . charHeight ; canvas . save ( ) ; canvas . translate ( x , y ) ; canvas . clipRect ( 0 , 0 , bridge . charWidth * ( onWideCharacter ? 2 : 1 ) , bridge . charHeight ) ; int metaState = bridge . getKeyHandler ( ) . getMetaState ( ) ; if ( y + bridge . charHeight < bridge . bitmap . getHeight ( ) ) { Bitmap underCursor = Bitmap . createBitmap ( bridge . bitmap , x , y , bridge . charWidth * ( onWideCharacter ? 2 : 1 ) , bridge . charHeight ) ; if ( metaState == 0 ) canvas . drawBitmap ( underCursor , 0 , 0 , cursorInversionPaint ) ; else canvas . drawBitmap ( underCursor , 0 , 0 , cursorMetaInversionPaint ) ; } else { canvas . drawPaint ( cursorPaint ) ; } final int deadKey = bridge . getKeyHandler ( ) . getDeadKey ( ) ; if ( deadKey != 0 ) { singleDeadKey [ 0 ] = ( char ) deadKey ; canvas . drawText ( singleDeadKey , 0 , 1 , 0 , 0 , cursorStrokePaint ) ; } canvas . concat ( scaleMatrix ) ; if ( ( metaState & TerminalKeyListener . OUR_SHIFT_ON ) != 0 ) canvas . drawPath ( shiftCursor , cursorStrokePaint ) ; else if ( ( metaState & TerminalKeyListener . OUR_SHIFT_LOCK ) != 0 ) canvas . drawPath ( shiftCursor , cursorInversionPaint ) ; if ( ( metaState & TerminalKeyListener . OUR_ALT_ON ) != 0 ) canvas . drawPath ( altCursor , cursorStrokePaint ) ; else if ( ( metaState & TerminalKeyListener . OUR_ALT_LOCK ) != 0 ) canvas . drawPath ( altCursor , cursorInversionPaint ) ; if ( ( metaState & TerminalKeyListener . OUR_CTRL_ON ) != 0 ) canvas . drawPath ( ctrlCursor , cursorStrokePaint ) ; else if ( ( metaState & TerminalKeyListener . OUR_CTRL_LOCK ) != 0 ) canvas . drawPath ( ctrlCursor , cursorInversionPaint ) ; canvas . restore ( ) ; } if ( terminalTextViewOverlay == null && bridge . isSelectingForCopy ( ) ) { SelectionArea area = bridge . getSelectionArea ( ) ; canvas . save ( ) ; canvas . clipRect ( area . getLeft ( ) * bridge . charWidth , area . getTop ( ) * bridge . charHeight , ( area . getRight ( ) + 1 ) * bridge . charWidth , ( area . getBottom ( ) + 1 ) * bridge . charHeight ) ; canvas . drawPaint ( cursorPaint ) ; canvas . restore ( ) ; } } } public void notifyUser ( String message ) { if ( !notifications ) return ; if ( notification != null ) { if ( lastNotification != null && lastNotification . equals ( message ) ) return ; notification . setText ( message ) ; notification . show ( ) ; } else { notification = Toast . makeText ( context , message , Toast . LENGTH_SHORT ) ; notification . show ( ) ; } lastNotification = message ; } public void forceSize ( int width , int height ) { bridge . resizeComputed ( width , height , getWidth ( ) , getHeight ( ) ) ; } public void setNotifications ( boolean value ) { notifications = value ; } @ Override public boolean onCheckIsTextEditor ( ) { return true ; } @ Override public InputConnection onCreateInputConnection ( EditorInfo outAttrs ) { outAttrs . imeOptions |= EditorInfo . IME_FLAG_NO_EXTRACT_UI | EditorInfo . IME_FLAG_NO_ENTER_ACTION | EditorInfo . IME_ACTION_NONE ; outAttrs . inputType = EditorInfo . TYPE_NULL | EditorInfo . TYPE_TEXT_VARIATION_PASSWORD | EditorInfo . TYPE_TEXT_VARIATION_VISIBLE_PASSWORD | EditorInfo . TYPE_TEXT_FLAG_NO_SUGGESTIONS ; return new BaseInputConnection ( this , false ) { @ Override public boolean deleteSurroundingText ( int leftLength , int rightLength ) { if ( rightLength == 0 && leftLength == 0 ) { return this . sendKeyEvent ( new KeyEvent ( KeyEvent . ACTION_DOWN , KeyEvent . KEYCODE_DEL ) ) ; } for ( int i = 0 ; i < leftLength ; i++ ) { this . sendKeyEvent ( new KeyEvent ( KeyEvent . ACTION_DOWN , KeyEvent . KEYCODE_DEL ) ) ; } return true ; } } ; } public void propagateConsoleText ( char [ ] rawText , int length ) { if ( mAccessibilityActive ) { synchronized ( mAccessibilityLock ) { mAccessibilityBuffer . append ( rawText , 0 , length ) ; } if ( mAccessibilityInitialized ) { if ( mEventSender != null ) { removeCallbacks ( mEventSender ) ; } else { mEventSender = new AccessibilityEventSender ( ) ; } postDelayed ( mEventSender , ACCESSIBILITY_EVENT_THRESHOLD ) ; } } ( ( Activity ) context ) . runOnUiThread ( new Runnable ( ) { @ Override public void run ( ) { if ( terminalTextViewOverlay != null ) { terminalTextViewOverlay . onBufferChanged ( ) ; } } } ) ; } private class AccessibilityEventSender implements Runnable { @ Override public void run ( ) { synchronized ( mAccessibilityLock ) { if ( mCodeMatcher == null ) { mCodeMatcher = mControlCodes . matcher ( mAccessibilityBuffer . toString ( ) ) ; } else { mCodeMatcher . reset ( mAccessibilityBuffer . toString ( ) ) ; } mAccessibilityBuffer . setLength ( 0 ) ; while ( mCodeMatcher . find ( ) ) { mCodeMatcher . appendReplacement ( mAccessibilityBuffer , " " ) ; } int i = mAccessibilityBuffer . indexOf ( BACKSPACE_CODE ) ; while ( i != -1 ) { mAccessibilityBuffer . replace ( i == 0 ? 0 : i - 1 , i + BACKSPACE_CODE . length ( ) , "" ) ; i = mAccessibilityBuffer . indexOf ( BACKSPACE_CODE ) ; } if ( mAccessibilityBuffer . length ( ) > 0 ) { AccessibilityEvent event = AccessibilityEvent . obtain ( AccessibilityEvent . TYPE_VIEW_TEXT_CHANGED ) ; event . setFromIndex ( 0 ) ; event . setAddedCount ( mAccessibilityBuffer . length ( ) ) ; event . getText ( ) . add ( mAccessibilityBuffer ) ; sendAccessibilityEventUnchecked ( event ) ; mAccessibilityBuffer . setLength ( 0 ) ; } } } } private class AccessibilityStateTester extends AsyncTask < Void , Void , Boolean > { @ Override protected Boolean doInBackground ( Void . . . params ) { final AccessibilityManager accessibility = ( AccessibilityManager ) context . getSystemService ( Context . ACCESSIBILITY_SERVICE ) ; if ( !accessibility . isEnabled ( ) ) { return Boolean . FALSE ; } final Intent screenReaderIntent = new Intent ( SCREENREADER_INTENT_ACTION ) ; screenReaderIntent . addCategory ( SCREENREADER_INTENT_CATEGORY ) ; final ContentResolver cr = context . getContentResolver ( ) ; final List < ResolveInfo > screenReaders = context . getPackageManager ( ) . queryIntentServices ( screenReaderIntent , 0 ) ; boolean foundScreenReader = false ; for ( ResolveInfo screenReader : screenReaders ) { final Cursor cursor = cr . query ( Uri . parse ( "content : + " . providers . StatusProvider" ) , null , null , null , null ) ; if ( cursor != null ) { try { if ( !cursor . moveToFirst ( ) ) { continue ; } final int status = cursor . getInt ( 0 ) ; if ( status == 1 ) { foundScreenReader = true ; break ; } } finally { cursor . close ( ) ; } } } if ( foundScreenReader ) { mControlCodes = Pattern . compile ( CONTROL_CODE_PATTERN ) ; } return foundScreenReader ; } @ Override protected void onPostExecute ( Boolean result ) { mAccessibilityActive = result ; mAccessibilityInitialized = true ; if ( result ) { mEventSender = new AccessibilityEventSender ( ) ; postDelayed ( mEventSender , ACCESSIBILITY_EVENT_THRESHOLD ) ; } else { synchronized ( mAccessibilityLock ) { mAccessibilityBuffer . setLength ( 0 ) ; mAccessibilityBuffer . trimToSize ( ) ; } } } } }