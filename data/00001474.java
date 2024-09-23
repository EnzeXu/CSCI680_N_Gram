public class FileAdaptorTap<TConfig, TInput, TOutput, OConfig, OInput, OOutput> extends AdaptorTap<TConfig, TInput, TOutput, OConfig, OInput, OOutput> implements FileType<TConfig> { @ConstructorProperties({"original", "processProvider", "configProvider"}) public FileAdaptorTap( Tap<OConfig, OInput, OOutput> original, Function<FlowProcess<? extends TConfig>, FlowProcess<? extends OConfig>> processProvider, Function<TConfig, OConfig> configProvider ) { super( original, processProvider, configProvider ); if( !( original instanceof FileType ) ) throw new IllegalArgumentException( "original Tap must be of type: " + FileType.class.getName() ); } protected FileType<TConfig> getFileOriginal() { return (FileType<TConfig>) getOriginal(); } @Override public boolean isDirectory( FlowProcess<? extends TConfig> flowProcess ) throws IOException { return getFileOriginal().isDirectory( (FlowProcess<? extends TConfig>) processProvider.apply( flowProcess ) ); } @Override public boolean isDirectory( TConfig conf ) throws IOException { return getFileOriginal().isDirectory( (TConfig) configProvider.apply( conf ) ); } @Override public String[] getChildIdentifiers( FlowProcess<? extends TConfig> flowProcess ) throws IOException { return getFileOriginal().getChildIdentifiers( (TConfig) processProvider.apply( flowProcess ) ); } @Override public String[] getChildIdentifiers( TConfig conf ) throws IOException { return getFileOriginal().getChildIdentifiers( (TConfig) configProvider.apply( conf ) ); } @Override public String[] getChildIdentifiers( FlowProcess<? extends TConfig> flowProcess, int depth, boolean fullyQualified ) throws IOException { return getFileOriginal().getChildIdentifiers( (TConfig) processProvider.apply( flowProcess ), depth, fullyQualified ); } @Override public String[] getChildIdentifiers( TConfig conf, int depth, boolean fullyQualified ) throws IOException { return getFileOriginal().getChildIdentifiers( (TConfig) configProvider.apply( conf ), depth, fullyQualified ); } @Override public long getSize( FlowProcess<? extends TConfig> flowProcess ) throws IOException { return getFileOriginal().getSize( (TConfig) processProvider.apply( flowProcess ) ); } @Override public long getSize( TConfig conf ) throws IOException { return getFileOriginal().getSize( (TConfig) configProvider.apply( conf ) ); } }