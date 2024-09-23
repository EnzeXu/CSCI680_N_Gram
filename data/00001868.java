public class PartitionTap extends BasePartitionTap<Properties, InputStream, OutputStream> { public static final String PART_NUM_PROPERTY = "cascading.local.tap.partition.seq"; @ConstructorProperties({"parent", "partition"}) public PartitionTap( Tap parent, Partition partition ) { this( parent, partition, OPEN_WRITES_THRESHOLD_DEFAULT ); } @ConstructorProperties({"parent", "partition", "openWritesThreshold"}) public PartitionTap( Tap parent, Partition partition, int openWritesThreshold ) { super( parent, partition, openWritesThreshold ); } @ConstructorProperties({"parent", "partition", "sinkMode"}) public PartitionTap( Tap parent, Partition partition, SinkMode sinkMode ) { super( parent, partition, sinkMode ); } @ConstructorProperties({"parent", "partition", "sinkMode", "keepParentOnDelete"}) public PartitionTap( Tap parent, Partition partition, SinkMode sinkMode, boolean keepParentOnDelete ) { this( parent, partition, sinkMode, keepParentOnDelete, OPEN_WRITES_THRESHOLD_DEFAULT ); } @ConstructorProperties({"parent", "partition", "sinkMode", "keepParentOnDelete", "openWritesThreshold"}) public PartitionTap( Tap parent, Partition partition, SinkMode sinkMode, boolean keepParentOnDelete, int openWritesThreshold ) { super( parent, partition, sinkMode, keepParentOnDelete, openWritesThreshold ); if( !( parent instanceof FileType ) ) throw new IllegalArgumentException( "parent Tap must be of type: " + FileType.class.getName() ); } @Override protected String getCurrentIdentifier( FlowProcess<? extends Properties> flowProcess ) { return null; } @Override public boolean deleteResource( Properties conf ) throws IOException { String[] childIdentifiers = ( (FileTap) parent ).getChildIdentifiers( conf, Integer.MAX_VALUE, false ); if( childIdentifiers.length == 0 ) return deleteParent( conf ); DirTap.deleteChildren( Paths.get( parent.getIdentifier() ), childIdentifiers ); return deleteParent( conf ); } private boolean deleteParent( Properties conf ) throws IOException { return keepParentOnDelete || parent.deleteResource( conf ); } @Override protected TupleEntrySchemeCollector createTupleEntrySchemeCollector( FlowProcess<? extends Properties> flowProcess, Tap parent, String path, long sequence ) throws IOException { if( sequence != -1 && flowProcess.getConfig() != null ) ( (LocalFlowProcess) FlowProcessWrapper.undelegate( flowProcess ) ).getConfig().setProperty( PART_NUM_PROPERTY, Long.toString( sequence ) ); if( parent instanceof TapWith ) return (TupleEntrySchemeCollector) ( (TapWith) parent ) .withChildIdentifier( path ) .withSinkMode( SinkMode.UPDATE ) .asTap().openForWrite( flowProcess ); TapFileOutputStream output = new TapFileOutputStream( parent, path, true ); return new TupleEntrySchemeCollector<Properties, OutputStream>( flowProcess, parent, output ); } @Override protected TupleEntrySchemeIterator createTupleEntrySchemeIterator( FlowProcess<? extends Properties> flowProcess, Tap parent, String path, InputStream input ) throws IOException { if( parent instanceof TapWith ) return (TupleEntrySchemeIterator) ( (TapWith) parent ) .withChildIdentifier( path ) .asTap().openForRead( flowProcess, input ); if( input == null ) input = new FileInputStream( path ); return new TupleEntrySchemeIterator( flowProcess, parent, parent.getScheme(), input, path ); } }