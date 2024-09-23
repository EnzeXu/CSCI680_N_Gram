public class TextViewSelect extends AppCompatEditText { public TextViewSelect(Context context) { this(context, null); } public TextViewSelect(Context context, AttributeSet attrs) { this(context, attrs, android.R.attr.textViewStyle); } public TextViewSelect(Context context, AttributeSet attrs, int defStyle) { super(context, attrs, defStyle); setFocusable(true); setFocusableInTouchMode(true); } @Override protected MovementMethod getDefaultMovementMethod() { return ArrowKeyMovementMethod.getInstance(); } @Override protected boolean getDefaultEditable() { return false; } @Override public void setText(CharSequence text, BufferType type) { super.setText(text, BufferType.EDITABLE); } }