public class ReadOnlyDialog extends WarningDialog { public ReadOnlyDialog(Context context) { super(context, R.string.show_read_only_warning); warning = context.getString(R.string.read_only_warning); if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) { warning = warning.concat("\n\n").concat(context.getString(R.string.read_only_kitkat_warning)); } } }