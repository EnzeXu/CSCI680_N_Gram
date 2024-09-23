public class Hadoop3TezFlowStep extends BaseFlowStep < TezConfiguration > { private static final Logger LOG = LoggerFactory . getLogger ( Hadoop3TezFlowStep . class ) ; private Map < String , LocalResource > allLocalResources = new HashMap < > ( ) ; private Map < Path , Path > syncPaths = new HashMap < > ( ) ; private Map < String , String > environment = new HashMap < > ( ) ; public Hadoop3TezFlowStep ( ElementGraph elementGraph , FlowNodeGraph flowNodeGraph ) { super ( elementGraph , flowNodeGraph ) ; } @ Override public Map < Object , Object > getConfigAsProperties ( ) { return HadoopUtil . createProperties ( getConfig ( ) ) ; } @ Override public TezConfiguration createInitializedConfig ( FlowProcess < TezConfiguration > flowProcess , TezConfiguration parentConfig ) { TezConfiguration stepConf = parentConfig == null ? new TezConfiguration ( ) : new TezConfiguration ( parentConfig ) ; Set < String > serializations = getFieldDeclaredSerializations ( Serialization . class ) ; TupleSerialization . setSerializations ( stepConf , serializations ) ; String versionString = Version . getRelease ( ) ; if ( versionString != null ) stepConf . set ( "cascading . version" , versionString ) ; stepConf . set ( CASCADING_FLOW_STEP_ID , getID ( ) ) ; stepConf . set ( "cascading . flow . step . num" , Integer . toString ( getOrdinal ( ) ) ) ; String flowStagingPath = ( ( Hadoop3TezFlow ) getFlow ( ) ) . getFlowStagingPath ( ) ; List < String > classPath = ( ( Hadoop3TezFlow ) getFlow ( ) ) . getClassPath ( ) ; Map < String , LocalResource > dagResources = new HashMap < > ( ) ; if ( !classPath . isEmpty ( ) ) { Map < Path , Path > dagClassPath = addToClassPath ( stepConf , flowStagingPath , null , classPath , FILE , dagResources , null ) ; syncPaths . putAll ( dagClassPath ) ; } String appJarPath = stepConf . get ( AppProps . APP_JAR_PATH ) ; if ( appJarPath != null ) { List < String > classpath = singletonList ( appJarPath ) ; Map < Path , Path > pathMap = addToClassPath ( stepConf , flowStagingPath , null , classpath , ARCHIVE , dagResources , environment ) ; syncPaths . putAll ( pathMap ) ; String fileName = new File ( appJarPath ) . getName ( ) ; stepConf . set ( TezConfiguration . TEZ_CLUSTER_ADDITIONAL_CLASSPATH_PREFIX , "$PWD/" + fileName + "/ : $PWD/" + fileName + "/classes/ : $PWD/" + fileName + "/lib/* : " ) ; } allLocalResources . putAll ( dagResources ) ; initFromStepConfigDef ( stepConf ) ; return stepConf ; } @ Override protected FlowStepJob createFlowStepJob ( ClientState clientState , FlowProcess < TezConfiguration > flowProcess , TezConfiguration initializedStepConfig ) { DAG dag = createDAG ( flowProcess , initializedStepConfig ) ; return new Hadoop3TezFlowStepJob ( clientState , this , initializedStepConfig , dag ) ; } private DAG createDAG ( FlowProcess < TezConfiguration > flowProcess , TezConfiguration initializedConfig ) { FlowNodeGraph nodeGraph = getFlowNodeGraph ( ) ; Map < FlowNode , Vertex > vertexMap = new HashMap < > ( ) ; DAG dag = DAG . create ( getStepDisplayName ( initializedConfig . getInt ( "cascading . display . id . truncate" , Util . ID_LENGTH ) ) ) ; dag . addTaskLocalFiles ( allLocalResources ) ; Iterator < FlowNode > iterator = nodeGraph . getOrderedTopologicalIterator ( ) ; while ( iterator . hasNext ( ) ) { FlowNode flowNode = iterator . next ( ) ; Vertex vertex = createVertex ( flowProcess , initializedConfig , flowNode ) ; dag . addVertex ( vertex ) ; vertexMap . put ( flowNode , vertex ) ; } LinkedList < ProcessEdge > processedEdges = new LinkedList < > ( ) ; for ( ProcessEdge processEdge : nodeGraph . edgeSet ( ) ) { if ( processedEdges . contains ( processEdge ) ) continue ; FlowNode edgeTargetFlowNode = nodeGraph . getEdgeTarget ( processEdge ) ; FlowElement flowElement = processEdge . getFlowElement ( ) ; List < FlowNode > sourceNodes = nodeGraph . getElementSourceProcesses ( flowElement ) ; EdgeProperty edgeProperty = createEdgeProperty ( initializedConfig , processEdge ) ; Vertex targetVertex = vertexMap . get ( edgeTargetFlowNode ) ; if ( sourceNodes . size ( ) == 1 || flowElement instanceof CoGroup || flowElement instanceof Boundary ) { FlowNode edgeSourceFlowNode = nodeGraph . getEdgeSource ( processEdge ) ; Vertex sourceVertex = vertexMap . get ( edgeSourceFlowNode ) ; LOG . debug ( "adding edge between : { } and { } " , sourceVertex , targetVertex ) ; dag . addEdge ( Edge . create ( sourceVertex , targetVertex , edgeProperty ) ) ; } else if ( flowElement instanceof GroupBy || flowElement instanceof Merge ) { List < String > sourceVerticesIDs = new ArrayList < > ( ) ; List < Vertex > sourceVertices = new ArrayList < > ( ) ; for ( FlowNode edgeSourceFlowNode : sourceNodes ) { sourceVerticesIDs . add ( edgeSourceFlowNode . getID ( ) ) ; sourceVertices . add ( vertexMap . get ( edgeSourceFlowNode ) ) ; processedEdges . add ( nodeGraph . getEdge ( edgeSourceFlowNode , edgeTargetFlowNode ) ) ; } VertexGroup vertexGroup = dag . createVertexGroup ( edgeTargetFlowNode . getID ( ) , sourceVertices . toArray ( new Vertex [ sourceVertices . size ( ) ] ) ) ; String inputClassName = flowElement instanceof Group ? OrderedGroupedMergedKVInput . class . getName ( ) : ConcatenatedMergedKeyValueInput . class . getName ( ) ; InputDescriptor inputDescriptor = InputDescriptor . create ( inputClassName ) . setUserPayload ( edgeProperty . getEdgeDestination ( ) . getUserPayload ( ) ) ; String type = ( ( Splice ) flowElement ) . isMerge ( ) ? "merged" : "grouped" ; LOG . info ( "adding { } edge between : { } and { } " , type , Util . join ( sourceVerticesIDs , " , " ) , targetVertex . getName ( ) ) ; dag . addEdge ( GroupInputEdge . create ( vertexGroup , targetVertex , edgeProperty , inputDescriptor ) ) ; } else { throw new UnsupportedOperationException ( "can't make edge for : " + flowElement ) ; } } return dag ; } private EdgeProperty createEdgeProperty ( TezConfiguration config , ProcessEdge processEdge ) { FlowElement flowElement = processEdge . getFlowElement ( ) ; EdgeValues edgeValues = new EdgeValues ( new TezConfiguration ( config ) , processEdge ) ; edgeValues . keyClassName = KeyTuple . class . getName ( ) ; edgeValues . valueClassName = ValueTuple . class . getName ( ) ; edgeValues . keyComparatorClassName = TupleComparator . class . getName ( ) ; edgeValues . keyPartitionerClassName = TuplePartitioner . class . getName ( ) ; edgeValues . outputClassName = null ; edgeValues . inputClassName = null ; edgeValues . movementType = null ; edgeValues . sourceType = null ; edgeValues . schedulingType = null ; if ( flowElement instanceof Group ) applyGroup ( edgeValues ) ; else if ( ( flowElement instanceof Boundary || flowElement instanceof Merge ) && processEdge . getSinkAnnotations ( ) . contains ( StreamMode . Accumulated ) ) applyBoundaryMergeAccumulated ( edgeValues ) ; else if ( flowElement instanceof Boundary || flowElement instanceof Merge ) applyBoundaryMerge ( edgeValues ) ; else throw new IllegalStateException ( "unsupported flow element : " + flowElement . getClass ( ) . getCanonicalName ( ) ) ; applyEdgeAnnotations ( processEdge , edgeValues ) ; return createEdgeProperty ( edgeValues ) ; } private void applyEdgeAnnotations ( ProcessEdge processEdge , EdgeValues edgeValues ) { processEdge . addEdgeAnnotation ( edgeValues . movementType ) ; processEdge . addEdgeAnnotation ( edgeValues . sourceType ) ; processEdge . addEdgeAnnotation ( edgeValues . schedulingType ) ; } private EdgeValues applyBoundaryMerge ( EdgeValues edgeValues ) { edgeValues . outputClassName = UnorderedPartitionedKVOutput . class . getName ( ) ; edgeValues . inputClassName = UnorderedKVInput . class . getName ( ) ; edgeValues . movementType = EdgeProperty . DataMovementType . SCATTER_GATHER ; edgeValues . sourceType = EdgeProperty . DataSourceType . PERSISTED ; edgeValues . schedulingType = EdgeProperty . SchedulingType . SEQUENTIAL ; return edgeValues ; } private EdgeValues applyBoundaryMergeAccumulated ( EdgeValues edgeValues ) { edgeValues . outputClassName = UnorderedKVOutput . class . getName ( ) ; edgeValues . inputClassName = UnorderedKVInput . class . getName ( ) ; edgeValues . movementType = EdgeProperty . DataMovementType . BROADCAST ; edgeValues . sourceType = EdgeProperty . DataSourceType . PERSISTED ; edgeValues . schedulingType = EdgeProperty . SchedulingType . SEQUENTIAL ; return edgeValues ; } private EdgeValues applyGroup ( EdgeValues edgeValues ) { Group group = ( Group ) edgeValues . flowElement ; if ( group . isSortReversed ( ) ) edgeValues . keyComparatorClassName = ReverseTupleComparator . class . getName ( ) ; int ordinal = getFirst ( edgeValues . ordinals ) ; addComparators ( edgeValues . config , "cascading . group . comparator" , group . getKeySelectors ( ) , edgeValues . getResolvedKeyFieldsMap ( ) . get ( ordinal ) ) ; if ( !group . isGroupBy ( ) ) { edgeValues . outputClassName = OrderedPartitionedKVOutput . class . getName ( ) ; edgeValues . inputClassName = OrderedGroupedKVInput . class . getName ( ) ; edgeValues . movementType = EdgeProperty . DataMovementType . SCATTER_GATHER ; edgeValues . sourceType = EdgeProperty . DataSourceType . PERSISTED ; edgeValues . schedulingType = EdgeProperty . SchedulingType . SEQUENTIAL ; } else { addComparators ( edgeValues . config , "cascading . sort . comparator" , group . getSortingSelectors ( ) , edgeValues . getResolvedSortFieldsMap ( ) . get ( ordinal ) ) ; edgeValues . outputClassName = OrderedPartitionedKVOutput . class . getName ( ) ; edgeValues . inputClassName = OrderedGroupedKVInput . class . getName ( ) ; edgeValues . movementType = EdgeProperty . DataMovementType . SCATTER_GATHER ; edgeValues . sourceType = EdgeProperty . DataSourceType . PERSISTED ; edgeValues . schedulingType = EdgeProperty . SchedulingType . SEQUENTIAL ; } if ( group . isSorted ( ) ) { edgeValues . keyClassName = TuplePair . class . getName ( ) ; edgeValues . keyPartitionerClassName = GroupingSortingPartitioner . class . getName ( ) ; if ( group . isSortReversed ( ) ) edgeValues . keyComparatorClassName = ReverseGroupingSortingComparator . class . getName ( ) ; else edgeValues . keyComparatorClassName = GroupingSortingComparator . class . getName ( ) ; } return edgeValues ; } private EdgeProperty createEdgeProperty ( EdgeValues edgeValues ) { TezConfiguration outputConfig = new TezConfiguration ( edgeValues . getConfig ( ) ) ; outputConfig . set ( "cascading . node . sink" , FlowElements . id ( edgeValues . getFlowElement ( ) ) ) ; outputConfig . set ( "cascading . node . ordinals" , Util . join ( edgeValues . getOrdinals ( ) , " , " ) ) ; addFields ( outputConfig , "cascading . node . key . fields" , edgeValues . getResolvedKeyFieldsMap ( ) ) ; addFields ( outputConfig , "cascading . node . sort . fields" , edgeValues . getResolvedSortFieldsMap ( ) ) ; addFields ( outputConfig , "cascading . node . value . fields" , edgeValues . getResolvedValueFieldsMap ( ) ) ; UserPayload outputPayload = createIntermediatePayloadOutput ( outputConfig , edgeValues ) ; TezConfiguration inputConfig = new TezConfiguration ( edgeValues . getConfig ( ) ) ; inputConfig . set ( "cascading . node . source" , FlowElements . id ( edgeValues . getFlowElement ( ) ) ) ; inputConfig . set ( "cascading . node . ordinals" , Util . join ( edgeValues . getOrdinals ( ) , " , " ) ) ; addFields ( inputConfig , "cascading . node . key . fields" , edgeValues . getResolvedKeyFieldsMap ( ) ) ; addFields ( inputConfig , "cascading . node . sort . fields" , edgeValues . getResolvedSortFieldsMap ( ) ) ; addFields ( inputConfig , "cascading . node . value . fields" , edgeValues . getResolvedValueFieldsMap ( ) ) ; UserPayload inputPayload = createIntermediatePayloadInput ( inputConfig , edgeValues ) ; return EdgeProperty . create ( edgeValues . getMovementType ( ) , edgeValues . getSourceType ( ) , edgeValues . getSchedulingType ( ) , OutputDescriptor . create ( edgeValues . getOutputClassName ( ) ) . setUserPayload ( outputPayload ) , InputDescriptor . create ( edgeValues . getInputClassName ( ) ) . setUserPayload ( inputPayload ) ) ; } private UserPayload createIntermediatePayloadOutput ( TezConfiguration config , EdgeValues edgeValues ) { config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_KEY_CLASS , edgeValues . keyClassName ) ; config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_VALUE_CLASS , edgeValues . valueClassName ) ; config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_KEY_COMPARATOR_CLASS , edgeValues . keyComparatorClassName ) ; config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_PARTITIONER_CLASS , edgeValues . keyPartitionerClassName ) ; setWorkingDirectory ( config ) ; return getPayload ( config ) ; } private UserPayload createIntermediatePayloadInput ( TezConfiguration config , EdgeValues edgeValues ) { config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_KEY_CLASS , edgeValues . keyClassName ) ; config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_VALUE_CLASS , edgeValues . valueClassName ) ; config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_KEY_COMPARATOR_CLASS , edgeValues . keyComparatorClassName ) ; config . set ( TezRuntimeConfiguration . TEZ_RUNTIME_PARTITIONER_CLASS , edgeValues . keyPartitionerClassName ) ; setWorkingDirectory ( config ) ; return getPayload ( config ) ; } private static void setWorkingDirectory ( Configuration conf ) { String name = conf . get ( JobContext . WORKING_DIR ) ; if ( name != null ) return ; try { Path dir = FileSystem . get ( conf ) . getWorkingDirectory ( ) ; conf . set ( JobContext . WORKING_DIR , dir . toString ( ) ) ; } catch ( IOException exception ) { throw new RuntimeException ( exception ) ; } } public Vertex createVertex ( FlowProcess < TezConfiguration > flowProcess , TezConfiguration initializedConfig , FlowNode flowNode ) { JobConf conf = new JobConf ( initializedConfig ) ; addInputOutputMapping ( conf , flowNode ) ; conf . setBoolean ( "mapred . used . genericoptionsparser" , true ) ; Map < String , LocalResource > taskLocalResources = new HashMap < > ( ) ; Map < FlowElement , Configuration > sourceConfigs = initFromSources ( flowNode , flowProcess , conf , taskLocalResources ) ; Map < FlowElement , Configuration > sinkConfigs = initFromSinks ( flowNode , flowProcess , conf ) ; initFromTraps ( flowNode , flowProcess , conf ) ; initFromNodeConfigDef ( flowNode , conf ) ; setLocalMode ( initializedConfig , conf , null ) ; conf . set ( "cascading . flow . node . num" , Integer . toString ( flowNode . getOrdinal ( ) ) ) ; HadoopUtil . setIsInflow ( conf ) ; int parallelism = getParallelism ( flowNode , conf ) ; if ( parallelism == 0 ) throw new FlowException ( getName ( ) , "the default number of gather partitions must be set , see cascading . flow . FlowRuntimeProps" ) ; flowNode . addProcessAnnotation ( FlowRuntimeProps . GATHER_PARTITIONS , Integer . toString ( parallelism ) ) ; Vertex vertex = newVertex ( flowNode , conf , parallelism ) ; if ( !taskLocalResources . isEmpty ( ) ) vertex . addTaskLocalFiles ( taskLocalResources ) ; for ( FlowElement flowElement : sourceConfigs . keySet ( ) ) { if ( ! ( flowElement instanceof Tap ) ) continue ; Configuration sourceConf = sourceConfigs . get ( flowElement ) ; if ( sourceConf . get ( "mapred . mapper . new-api" ) == null ) HadoopUtil . setNewApi ( sourceConf , sourceConf . get ( "mapred . input . format . class" , sourceConf . get ( "mapreduce . job . inputformat . class" ) ) ) ; MRInput . MRInputConfigBuilder configBuilder = MRInput . createConfigBuilder ( sourceConf , null ) ; if ( conf . get ( FlowRuntimeProps . COMBINE_SPLITS ) != null ) configBuilder . groupSplits ( conf . getBoolean ( FlowRuntimeProps . COMBINE_SPLITS , true ) ) ; if ( !CompositeTaps . unwindNarrow ( PartitionTap . class , ( Tap ) flowElement ) . isEmpty ( ) ) configBuilder . groupSplits ( false ) ; DataSourceDescriptor dataSourceDescriptor = configBuilder . build ( ) ; vertex . addDataSource ( FlowElements . id ( flowElement ) , dataSourceDescriptor ) ; } for ( FlowElement flowElement : sinkConfigs . keySet ( ) ) { if ( ! ( flowElement instanceof Tap ) ) continue ; Configuration sinkConf = sinkConfigs . get ( flowElement ) ; Class outputFormatClass ; String outputPath ; String formatClassName = sinkConf . get ( "mapred . output . format . class" , sinkConf . get ( "mapreduce . job . outputformat . class" ) ) ; if ( formatClassName == null ) { outputFormatClass = TextOutputFormat . class ; outputPath = Hfs . getTempPath ( sinkConf ) . toString ( ) ; } else { outputFormatClass = Util . loadClass ( formatClassName ) ; outputPath = getOutputPath ( sinkConf ) ; } if ( outputPath == null && getOutputPath ( sinkConf ) == null && isFileOutputFormat ( outputFormatClass ) ) outputPath = Hfs . getTempPath ( sinkConf ) . toString ( ) ; MROutput . MROutputConfigBuilder configBuilder = MROutput . createConfigBuilder ( sinkConf , outputFormatClass , outputPath ) ; DataSinkDescriptor dataSinkDescriptor = configBuilder . build ( ) ; vertex . addDataSink ( FlowElements . id ( flowElement ) , dataSinkDescriptor ) ; } addRemoteDebug ( flowNode , vertex ) ; addRemoteProfiling ( flowNode , vertex ) ; if ( vertex . getTaskLaunchCmdOpts ( ) != null ) flowNode . addProcessAnnotation ( TezConfiguration . TEZ_TASK_LAUNCH_CMD_OPTS , vertex . getTaskLaunchCmdOpts ( ) ) ; return vertex ; } protected String getOutputPath ( Configuration sinkConf ) { return sinkConf . get ( "mapred . output . dir" , sinkConf . get ( "mapreduce . output . fileoutputformat . outputdir" ) ) ; } protected boolean isFileOutputFormat ( Class outputFormatClass ) { return org . apache . hadoop . mapred . FileOutputFormat . class . isAssignableFrom ( outputFormatClass ) || org . apache . hadoop . mapreduce . lib . output . FileOutputFormat . class . isAssignableFrom ( outputFormatClass ) ; } protected int getParallelism ( FlowNode flowNode , JobConf conf ) { HashSet < Tap > sourceStreamedTaps = new HashSet < > ( flowNode . getSourceTaps ( ) ) ; sourceStreamedTaps . removeAll ( flowNode . getSourceElements ( StreamMode . Accumulated ) ) ; if ( sourceStreamedTaps . size ( ) != 0 ) return -1 ; int parallelism = Integer . MAX_VALUE ; for ( Tap tap : flowNode . getSinkTaps ( ) ) { int numSinkParts = tap . getScheme ( ) . getNumSinkParts ( ) ; if ( numSinkParts == 0 ) continue ; if ( parallelism != Integer . MAX_VALUE ) LOG . info ( "multiple sink taps in flow node declaring numSinkParts , choosing lowest value . see cascading . flow . FlowRuntimeProps for broader control . " ) ; parallelism = Math . min ( parallelism , numSinkParts ) ; } if ( parallelism != Integer . MAX_VALUE ) return parallelism ; return conf . getInt ( FlowRuntimeProps . GATHER_PARTITIONS , 0 ) ; } private void addInputOutputMapping ( JobConf conf , FlowNode flowNode ) { FlowNodeGraph flowNodeGraph = getFlowNodeGraph ( ) ; Set < ProcessEdge > incomingEdges = flowNodeGraph . incomingEdgesOf ( flowNode ) ; for ( ProcessEdge processEdge : incomingEdges ) conf . set ( "cascading . node . source . " + processEdge . getFlowElementID ( ) , processEdge . getSourceProcessID ( ) ) ; Set < ProcessEdge > outgoingEdges = flowNodeGraph . outgoingEdgesOf ( flowNode ) ; for ( ProcessEdge processEdge : outgoingEdges ) conf . set ( "cascading . node . sink . " + processEdge . getFlowElementID ( ) , processEdge . getSinkProcessID ( ) ) ; } protected Map < FlowElement , Configuration > initFromSources ( FlowNode flowNode , FlowProcess < TezConfiguration > flowProcess , Configuration conf , Map < String , LocalResource > taskLocalResources ) { Set < ? extends FlowElement > accumulatedSources = flowNode . getSourceElements ( StreamMode . Accumulated ) ; for ( FlowElement element : accumulatedSources ) { if ( element instanceof Tap ) { JobConf current = new JobConf ( conf ) ; Tap tap = ( Tap ) element ; if ( tap . getIdentifier ( ) == null ) throw new IllegalStateException ( "tap may not have null identifier : " + tap . toString ( ) ) ; tap . sourceConfInit ( flowProcess , current ) ; Collection < String > paths = current . getStringCollection ( CASCADING_LOCAL_RESOURCES + Tap . id ( tap ) ) ; if ( !paths . isEmpty ( ) ) { String flowStagingPath = ( ( Hadoop3TezFlow ) getFlow ( ) ) . getFlowStagingPath ( ) ; String resourceSubPath = Tap . id ( tap ) ; Map < Path , Path > pathMap = TezUtil . addToClassPath ( current , flowStagingPath , resourceSubPath , paths , LocalResourceType . FILE , taskLocalResources , null ) ; current . setStrings ( CASCADING_REMOTE_RESOURCES + Tap . id ( tap ) , taskLocalResources . keySet ( ) . toArray ( new String [ taskLocalResources . size ( ) ] ) ) ; allLocalResources . putAll ( taskLocalResources ) ; syncPaths . putAll ( pathMap ) ; } Map < String , String > map = flowProcess . diffConfigIntoMap ( new TezConfiguration ( conf ) , new TezConfiguration ( current ) ) ; conf . set ( "cascading . node . accumulated . source . conf . " + Tap . id ( tap ) , pack ( map , conf ) ) ; setLocalMode ( conf , current , tap ) ; } } Set < FlowElement > sources = new HashSet < > ( flowNode . getSourceElements ( ) ) ; sources . removeAll ( accumulatedSources ) ; if ( sources . isEmpty ( ) ) throw new IllegalStateException ( "all sources marked as accumulated" ) ; Map < FlowElement , Configuration > configs = new HashMap < > ( ) ; for ( FlowElement element : sources ) { JobConf current = new JobConf ( conf ) ; String id = FlowElements . id ( element ) ; current . set ( "cascading . node . source" , id ) ; if ( element instanceof Tap ) { Tap tap = ( Tap ) element ; if ( tap . getIdentifier ( ) == null ) throw new IllegalStateException ( "tap may not have null identifier : " + tap . toString ( ) ) ; tap . sourceConfInit ( flowProcess , current ) ; setLocalMode ( conf , current , tap ) ; } configs . put ( element , current ) ; } return configs ; } protected Map < FlowElement , Configuration > initFromSinks ( FlowNode flowNode , FlowProcess < ? extends Configuration > flowProcess , Configuration conf ) { Set < FlowElement > sinks = flowNode . getSinkElements ( ) ; Map < FlowElement , Configuration > configs = new HashMap < > ( ) ; for ( FlowElement element : sinks ) { JobConf current = new JobConf ( conf ) ; if ( element instanceof Tap ) { Tap tap = ( Tap ) element ; if ( tap . getIdentifier ( ) == null ) throw new IllegalStateException ( "tap may not have null identifier : " + element . toString ( ) ) ; tap . sinkConfInit ( flowProcess , current ) ; setLocalMode ( conf , current , tap ) ; } String id = FlowElements . id ( element ) ; current . set ( "cascading . node . sink" , id ) ; configs . put ( element , current ) ; } return configs ; } private void initFromNodeConfigDef ( FlowNode flowNode , Configuration conf ) { initConfFromNodeConfigDef ( flowNode . getElementGraph ( ) , new ConfigurationSetter ( conf ) ) ; } private void initFromStepConfigDef ( Configuration conf ) { initConfFromStepConfigDef ( new ConfigurationSetter ( conf ) ) ; } protected void initFromTraps ( FlowNode flowNode , FlowProcess < ? extends Configuration > flowProcess , Configuration conf ) { Map < String , Tap > traps = flowNode . getTrapMap ( ) ; if ( !traps . isEmpty ( ) ) { JobConf trapConf = new JobConf ( conf ) ; for ( Tap tap : traps . values ( ) ) { tap . sinkConfInit ( flowProcess , trapConf ) ; setLocalMode ( conf , trapConf , tap ) ; } } } private Vertex newVertex ( FlowNode flowNode , Configuration conf , int parallelism ) { conf . set ( FlowNode . CASCADING_FLOW_NODE , pack ( flowNode , conf ) ) ; ProcessorDescriptor descriptor = ProcessorDescriptor . create ( FlowProcessor . class . getName ( ) ) ; descriptor . setUserPayload ( getPayload ( conf ) ) ; Vertex vertex = Vertex . create ( flowNode . getID ( ) , descriptor , parallelism ) ; if ( environment != null ) vertex . setTaskEnvironment ( environment ) ; return vertex ; } private UserPayload getPayload ( Configuration conf ) { try { return TezUtils . createUserPayloadFromConf ( conf ) ; } catch ( IOException exception ) { throw new CascadingException ( exception ) ; } } private String pack ( Object object , Configuration conf ) { try { return serializeBase64 ( object , conf , true ) ; } catch ( IOException exception ) { throw new FlowException ( "unable to pack object : " + object . getClass ( ) . getCanonicalName ( ) , exception ) ; } } @ Override public void clean ( TezConfiguration config ) { for ( Tap sink : getSinkTaps ( ) ) { if ( sink . isTemporary ( ) && ( getFlow ( ) . getFlowStats ( ) . isSuccessful ( ) || getFlow ( ) . getRunID ( ) == null ) ) { try { sink . deleteResource ( config ) ; } catch ( Exception exception ) { logWarn ( "unable to remove temporary file : " + sink , exception ) ; } } else { cleanTapMetaData ( config , sink ) ; } } for ( Tap tap : getTraps ( ) ) cleanTapMetaData ( config , tap ) ; } private void cleanTapMetaData ( TezConfiguration config , Tap tap ) { try { Hadoop18TapUtil . cleanupTapMetaData ( config , tap ) ; } catch ( IOException exception ) { } } public void syncArtifacts ( ) { Map < String , Long > timestamps = HadoopUtil . syncPaths ( getConfig ( ) , syncPaths , true ) ; for ( Map . Entry < String , Long > entry : timestamps . entrySet ( ) ) { LocalResource localResource = allLocalResources . get ( entry . getKey ( ) ) ; if ( localResource != null ) localResource . setTimestamp ( entry . getValue ( ) ) ; } } private void setLocalMode ( Configuration parent , JobConf current , Tap tap ) { if ( !HadoopUtil . isLocal ( current ) ) return ; if ( tap != null ) logInfo ( "tap forcing step to tez local mode : " + tap . getIdentifier ( ) ) ; HadoopUtil . setLocal ( parent ) ; } private void addRemoteDebug ( FlowNode flowNode , Vertex vertex ) { String value = System . getProperty ( "test . debug . node" , null ) ; if ( Util . isEmpty ( value ) ) return ; if ( !flowNode . getSourceElementNames ( ) . contains ( value ) && asInt ( value ) != flowNode . getOrdinal ( ) ) return ; LOG . warn ( "remote debugging enabled with property : { } , on node : { } , with node id : { } " , "test . debug . node" , value , flowNode . getID ( ) ) ; String opts = vertex . getTaskLaunchCmdOpts ( ) ; if ( opts == null ) opts = "" ; String address = System . getProperty ( "test . debug . address" , "localhost : 5005" ) . trim ( ) ; opts += " -agentlib : jdwp=transport=dt_socket , server=n , address=" + address + " , suspend=y" ; vertex . setTaskLaunchCmdOpts ( opts ) ; } private void addRemoteProfiling ( FlowNode flowNode , Vertex vertex ) { String value = System . getProperty ( "test . profile . node" , null ) ; if ( Util . isEmpty ( value ) ) return ; if ( !flowNode . getSourceElementNames ( ) . contains ( value ) && asInt ( value ) != flowNode . getOrdinal ( ) ) return ; LOG . warn ( "remote profiling enabled with property : { } , on node : { } , with node id : { } " , "test . profile . node" , value , flowNode . getID ( ) ) ; String opts = vertex . getTaskLaunchCmdOpts ( ) ; if ( opts == null ) opts = "" ; String path = System . getProperty ( "test . profile . path" , "/tmp/jfr/" ) ; if ( !path . endsWith ( "/" ) ) path += "/" ; LOG . warn ( "remote profiling property : { } , logging to path : { } " , "test . profile . path" , path ) ; opts += String . format ( " -XX : +UnlockCommercialFeatures -XX : +FlightRecorder -XX : FlightRecorderOptions=defaultrecording=true , dumponexit=true , dumponexitpath=%1$s%2$s , disk=true , repository=%1$s%2$s" , path , flowNode . getID ( ) ) ; vertex . setTaskLaunchCmdOpts ( opts ) ; } private int asInt ( String value ) { try { return Integer . parseInt ( value ) ; } catch ( NumberFormatException exception ) { return -1 ; } } public Map < String , LocalResource > getAllLocalResources ( ) { return allLocalResources ; } private static class EdgeValues { FlowElement flowElement ; TezConfiguration config ; Set < Integer > ordinals ; String keyClassName ; String valueClassName ; String keyComparatorClassName ; String keyPartitionerClassName ; String outputClassName ; String inputClassName ; EdgeProperty . DataMovementType movementType ; EdgeProperty . DataSourceType sourceType ; EdgeProperty . SchedulingType schedulingType ; Map < Integer , Fields > resolvedKeyFieldsMap ; Map < Integer , Fields > resolvedSortFieldsMap ; Map < Integer , Fields > resolvedValueFieldsMap ; private EdgeValues ( TezConfiguration config , ProcessEdge processEdge ) { this . config = config ; this . flowElement = processEdge . getFlowElement ( ) ; this . ordinals = processEdge . getSourceProvidedOrdinals ( ) ; this . resolvedKeyFieldsMap = processEdge . getResolvedKeyFields ( ) ; this . resolvedSortFieldsMap = processEdge . getResolvedSortFields ( ) ; this . resolvedValueFieldsMap = processEdge . getResolvedValueFields ( ) ; } public FlowElement getFlowElement ( ) { return flowElement ; } public TezConfiguration getConfig ( ) { return config ; } public Set getOrdinals ( ) { return ordinals ; } public String getKeyClassName ( ) { return keyClassName ; } public String getValueClassName ( ) { return valueClassName ; } public String getKeyComparatorClassName ( ) { return keyComparatorClassName ; } public String getKeyPartitionerClassName ( ) { return keyPartitionerClassName ; } public String getOutputClassName ( ) { return outputClassName ; } public String getInputClassName ( ) { return inputClassName ; } public EdgeProperty . DataMovementType getMovementType ( ) { return movementType ; } public EdgeProperty . DataSourceType getSourceType ( ) { return sourceType ; } public EdgeProperty . SchedulingType getSchedulingType ( ) { return schedulingType ; } public Map < Integer , Fields > getResolvedKeyFieldsMap ( ) { return resolvedKeyFieldsMap ; } public Map < Integer , Fields > getResolvedSortFieldsMap ( ) { return resolvedSortFieldsMap ; } public Map < Integer , Fields > getResolvedValueFieldsMap ( ) { return resolvedValueFieldsMap ; } } }