public class TezUtil { private static final Logger LOG = LoggerFactory.getLogger( TezUtil.class ); public static JobConf asJobConf( Configuration configuration ) { return new JobConf( configuration ); } public static TezConfiguration createTezConf( Map<Object, Object> properties, TezConfiguration defaultJobconf ) { TezConfiguration jobConf = defaultJobconf == null ? new TezConfiguration() : new TezConfiguration( defaultJobconf ); if( properties == null ) return jobConf; Set<Object> keys = new HashSet<Object>( properties.keySet() ); if( properties instanceof Properties ) keys.addAll( ( (Properties) properties ).stringPropertyNames() ); for( Object key : keys ) { Object value = properties.get( key ); if( value == null && properties instanceof Properties && key instanceof String ) value = ( (Properties) properties ).getProperty( (String) key ); if( value == null ) continue; if( value instanceof Class || value instanceof TezConfiguration ) continue; jobConf.set( key.toString(), value.toString() ); } return jobConf; } public static UserGroupInformation getCurrentUser() { try { return UserGroupInformation.getCurrentUser(); } catch( IOException exception ) { throw new CascadingException( "unable to get current user", exception ); } } public static String getEdgeSourceID( LogicalInput input, Configuration configuration ) { String id = configuration.get( "cascading.node.source" ); if( id == null ) throw new IllegalStateException( "no source id found: " + input.getClass().getName() ); return id; } public static String getEdgeSinkID( LogicalOutput output, Configuration configuration ) { String id = configuration.get( "cascading.node.sink" ); if( id == null ) throw new IllegalStateException( "no sink id found: " + output.getClass().getName() ); return id; } public static Configuration getInputConfiguration( LogicalInput input ) { try { if( input instanceof MergedLogicalInput ) input = (LogicalInput) Util.getFirst( ( (MergedLogicalInput) input ).getInputs() ); if( input instanceof MRInput ) return createConfFromByteString( parseMRInputPayload( ( (MRInput) input ).getContext().getUserPayload() ).getConfigurationBytes() ); if( input instanceof AbstractLogicalInput ) return createConfFromUserPayload( ( (AbstractLogicalInput) input ).getContext().getUserPayload() ); } catch( IOException exception ) { throw new FlowException( "unable to unpack payload", exception ); } throw new IllegalStateException( "unknown input type: " + input.getClass().getName() ); } public static Configuration getOutputConfiguration( LogicalOutput output ) { try { if( output instanceof MROutput ) return TezUtils.createConfFromUserPayload( ( (MROutput) output ).getContext().getUserPayload() ); if( output instanceof AbstractLogicalOutput ) return createConfFromUserPayload( ( (AbstractLogicalOutput) output ).getContext().getUserPayload() ); } catch( IOException exception ) { throw new FlowException( "unable to unpack payload", exception ); } throw new IllegalStateException( "unknown input type: " + output.getClass().getName() ); } public static void setSourcePathForSplit( MRInput input, MRReader reader, Configuration configuration ) { Path path = null; if( Util.returnInstanceFieldIfExistsSafe( input, "useNewApi" ) ) { org.apache.hadoop.mapreduce.InputSplit newInputSplit = (org.apache.hadoop.mapreduce.InputSplit) reader.getSplit(); if( newInputSplit instanceof org.apache.hadoop.mapreduce.lib.input.FileSplit ) path = ( (org.apache.hadoop.mapreduce.lib.input.FileSplit) newInputSplit ).getPath(); } else { org.apache.hadoop.mapred.InputSplit oldInputSplit = (org.apache.hadoop.mapred.InputSplit) reader.getSplit(); if( oldInputSplit instanceof org.apache.hadoop.mapred.FileSplit ) path = ( (org.apache.hadoop.mapred.FileSplit) oldInputSplit ).getPath(); } if( path != null ) configuration.set( FileType.CASCADING_SOURCE_PATH, path.toString() ); } public static Map<Path, Path> addToClassPath( Configuration config, String stagingRoot, String resourceSubPath, Collection<String> classpath, LocalResourceType resourceType, Map<String, LocalResource> localResources, Map<String, String> environment ) { if( classpath == null ) return null; Map<String, Path> localPaths = new HashMap<>(); Map<String, Path> remotePaths = new HashMap<>(); HadoopUtil.resolvePaths( config, classpath, stagingRoot, resourceSubPath, localPaths, remotePaths ); try { LocalFileSystem localFS = HadoopUtil.getLocalFS( config ); for( String fileName : localPaths.keySet() ) { Path artifact = localPaths.get( fileName ); Path remotePath = remotePaths.get( fileName ); if( remotePath == null ) remotePath = artifact; addResource( localResources, environment, fileName, localFS.getFileStatus( artifact ), remotePath, resourceType ); } FileSystem defaultFS = HadoopUtil.getDefaultFS( config ); for( String fileName : remotePaths.keySet() ) { Path artifact = remotePaths.get( fileName ); Path localPath = localPaths.get( fileName ); if( localPath != null ) continue; addResource( localResources, environment, fileName, defaultFS.getFileStatus( artifact ), artifact, resourceType ); } } catch( IOException exception ) { throw new FlowException( "unable to set remote resource paths", exception ); } return getCommonPaths( localPaths, remotePaths ); } protected static void addResource( Map<String, LocalResource> localResources, Map<String, String> environment, String fileName, FileStatus stats, Path fullPath, LocalResourceType type ) throws IOException { if( localResources.containsKey( fileName ) ) throw new FlowException( "duplicate filename added to classpath resources: " + fileName ); URL yarnUrlFromPath = ConverterUtils.getYarnUrlFromPath( fullPath ); long len = stats.getLen(); long modificationTime = stats.getModificationTime(); LocalResource resource = LocalResource.newInstance( yarnUrlFromPath, type, LocalResourceVisibility.APPLICATION, len, modificationTime ); if( type == LocalResourceType.PATTERN ) { String pattern = "(?:classes/|lib/).*"; resource.setPattern( pattern ); if( environment != null ) { String current = ""; current += PWD.$$() + File.separator + fileName + File.separator + "*" + CLASS_PATH_SEPARATOR; current += PWD.$$() + File.separator + fileName + File.separator + "lib" + File.separator + "*" + CLASS_PATH_SEPARATOR; current += PWD.$$() + File.separator + fileName + File.separator + "classes" + File.separator + "*" + CLASS_PATH_SEPARATOR; String classPath = environment.get( CLASSPATH.name() ); if( classPath == null ) classPath = ""; else if( !classPath.startsWith( CLASS_PATH_SEPARATOR ) ) classPath += CLASS_PATH_SEPARATOR; classPath += current; LOG.info( "adding to cluster side classpath: {} ", classPath ); environment.put( CLASSPATH.name(), classPath ); } } localResources.put( fileName, resource ); } public static void setMRProperties( ProcessorContext context, Configuration config, boolean isMapperOutput ) { TaskAttemptID taskAttemptId = org.apache.tez.mapreduce.hadoop.mapreduce.TaskAttemptContextImpl .createMockTaskAttemptID( context.getApplicationId().getClusterTimestamp(), context.getTaskVertexIndex(), context.getApplicationId().getId(), context.getTaskIndex(), context.getTaskAttemptNumber(), isMapperOutput ); config.set( JobContext.TASK_ATTEMPT_ID, taskAttemptId.toString() ); config.set( JobContext.TASK_ID, taskAttemptId.getTaskID().toString() ); config.setBoolean( JobContext.TASK_ISMAP, isMapperOutput ); config.setInt( JobContext.TASK_PARTITION, taskAttemptId.getTaskID().getId() ); } }