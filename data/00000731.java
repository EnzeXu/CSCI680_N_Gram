public class LongEditTextPreferenceFragment extends EditTextPreferenceDialogFragmentCompat { private final LongEditTextPreference preference ; public LongEditTextPreferenceFragment ( LongEditTextPreference preference ) { this . preference = preference ; final Bundle b = new Bundle ( ) ; b . putString ( ARG_KEY , preference . getKey ( ) ) ; setArguments ( b ) ; } @ NonNull @ Override public Dialog onCreateDialog ( Bundle savedInstanceState ) { return super . onCreateDialog ( savedInstanceState ) ; } }