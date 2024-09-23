public class LongEditTextPreference extends EditTextPreference { public LongEditTextPreference ( Context context , AttributeSet attrs , int defStyleAttr , int defStyleRes ) { super ( context , attrs , defStyleAttr , defStyleRes ) ; } public LongEditTextPreference ( Context context , AttributeSet attrs , int defStyleAttr ) { super ( context , attrs , defStyleAttr ) ; } public LongEditTextPreference ( Context context , AttributeSet attrs ) { super ( context , attrs ) ; } public LongEditTextPreference ( Context context ) { super ( context ) ; } @ Override protected String getPersistedString ( String defaultReturnValue ) { return String . valueOf ( getPersistedLong ( -1 ) ) ; } @ Override protected boolean persistString ( String value ) { try { return persistLong ( Long . valueOf ( value ) ) ; } catch ( NumberFormatException e ) { Toast . makeText ( getContext ( ) , R . string . error_rounds_not_number , Toast . LENGTH_LONG ) . show ( ) ; } return false ; } }