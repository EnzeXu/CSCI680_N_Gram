public class GroupEmptyView extends RelativeLayout { public GroupEmptyView(Context context) { this(context, null); } public GroupEmptyView(Context context, AttributeSet attrs) { super(context, attrs); inflate(context); } private void inflate(Context context) { LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE); inflater.inflate(R.layout.group_empty, this); } }