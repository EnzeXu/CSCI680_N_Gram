public class PasswordTextViewSelect extends TextViewSelect { public PasswordTextViewSelect ( Context context , AttributeSet attrs , int defStyle ) { super ( context , attrs , defStyle ) ; } public PasswordTextViewSelect ( Context context , AttributeSet attrs ) { super ( context , attrs ) ; } public PasswordTextViewSelect ( Context context ) { super ( context ) ; } private Typeface getTypeface ( Typeface tf ) { Typeface tfOverride = TypefaceFactory . getTypeface ( getContext ( ) , "fonts/DejaVuSansMono . ttf" ) ; if ( tfOverride != null ) { return tfOverride ; } return tf ; } @ Override public void setTypeface ( Typeface tf , int style ) { super . setTypeface ( getTypeface ( tf ) , style ) ; } @ Override public void setTypeface ( Typeface tf ) { super . setTypeface ( getTypeface ( tf ) ) ; } }