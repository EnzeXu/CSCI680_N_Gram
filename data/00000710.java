public class GroupEditActivity extends AppCompatActivity { public static final String KEY_NAME = "name" ; public static final String KEY_ICON_ID = "icon_id" ; private int mSelectedIconID ; public static void Launch ( Activity act ) { Intent i = new Intent ( act , GroupEditActivity . class ) ; act . startActivityForResult ( i , 0 ) ; } @ Override protected void onCreate ( Bundle savedInstanceState ) { super . onCreate ( savedInstanceState ) ; setContentView ( R . layout . group_edit ) ; setTitle ( R . string . add_group_title ) ; ImageButton iconButton = ( ImageButton ) findViewById ( R . id . icon_button ) ; iconButton . setOnClickListener ( new View . OnClickListener ( ) { public void onClick ( View v ) { IconPickerActivity . Launch ( GroupEditActivity . this ) ; } } ) ; Button okButton = ( Button ) findViewById ( R . id . ok ) ; okButton . setOnClickListener ( new View . OnClickListener ( ) { public void onClick ( View v ) { TextView nameField = ( TextView ) findViewById ( R . id . group_name ) ; String name = nameField . getText ( ) . toString ( ) ; if ( name . length ( ) > 0 ) { final Intent intent = new Intent ( ) ; intent . putExtra ( KEY_NAME , name ) ; intent . putExtra ( KEY_ICON_ID , mSelectedIconID ) ; setResult ( Activity . RESULT_OK , intent ) ; finish ( ) ; } else { Toast . makeText ( GroupEditActivity . this , R . string . error_no_name , Toast . LENGTH_LONG ) . show ( ) ; } } } ) ; Button cancel = ( Button ) findViewById ( R . id . cancel ) ; cancel . setOnClickListener ( new View . OnClickListener ( ) { public void onClick ( View v ) { final Intent intent = new Intent ( ) ; setResult ( Activity . RESULT_CANCELED , intent ) ; finish ( ) ; } } ) ; } @ Override protected void onActivityResult ( int requestCode , int resultCode , Intent data ) { super . onActivityResult ( requestCode , resultCode , data ) ; switch ( resultCode ) { case EntryEditActivity . RESULT_OK_ICON_PICKER : mSelectedIconID = data . getExtras ( ) . getInt ( IconPickerActivity . KEY_ICON_ID ) ; ImageButton currIconButton = ( ImageButton ) findViewById ( R . id . icon_button ) ; currIconButton . setImageResource ( Icons . iconToResId ( mSelectedIconID ) ) ; break ; case Activity . RESULT_CANCELED : default : break ; } } }