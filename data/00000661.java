public class VolumePreferenceFragment extends PreferenceDialogFragmentCompat { private SeekBar mVolumeBar ; public VolumePreferenceFragment ( ) { } public static VolumePreferenceFragment newInstance ( Preference preference ) { VolumePreferenceFragment fragment = new VolumePreferenceFragment ( ) ; Bundle bundle = new Bundle ( 1 ) ; bundle . putString ( "key" , preference . getKey ( ) ) ; fragment . setArguments ( bundle ) ; return fragment ; } @ Override protected void onBindDialogView ( View view ) { super . onBindDialogView ( view ) ; mVolumeBar = view . findViewById ( R . id . volume_bar ) ; Integer volumeLevel = null ; DialogPreference preference = getPreference ( ) ; if ( preference instanceof VolumePreference ) { volumeLevel = ( ( VolumePreference ) preference ) . getVolume ( ) ; } if ( volumeLevel != null ) { mVolumeBar . setProgress ( volumeLevel ) ; } } @ Override public void onDialogClosed ( boolean positiveResult ) { if ( positiveResult ) { int volumeLevel = mVolumeBar . getProgress ( ) ; DialogPreference preference = getPreference ( ) ; if ( preference instanceof VolumePreference ) { VolumePreference volumePreference = ( VolumePreference ) preference ; if ( volumePreference . callChangeListener ( volumeLevel ) ) { volumePreference . setVolume ( volumeLevel ) ; } } } } }