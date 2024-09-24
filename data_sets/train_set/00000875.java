public class GroupViewOnlyView extends RelativeLayout { public GroupViewOnlyView(Context context) { this(context, null); } public GroupViewOnlyView(Context context, AttributeSet attrs) { super(context, attrs); inflate(context); } private void inflate(Context context) { LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE); inflater.inflate(R.layout.group_add_entry, this); View addGroup = findViewById(R.id.add_group); addGroup.setVisibility(INVISIBLE); View addEntry = findViewById(R.id.add_entry); addEntry.setVisibility(INVISIBLE); View divider2 = findViewById(R.id.divider2); divider2.setVisibility(INVISIBLE); View list = findViewById(R.id.group_list); LayoutParams lp = (RelativeLayout.LayoutParams) list.getLayoutParams(); lp.addRule(ALIGN_PARENT_BOTTOM, TRUE); } }