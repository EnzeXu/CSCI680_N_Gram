public class Partitions extends GraphResult { private RulePartitioner rulePartitioner; private final GraphPartitioner graphPartitioner; private final ElementGraph beginGraph; private final Map<ElementGraph, EnumMultiMap> annotatedSubGraphs; private ElementGraph contractedGraph; private List<Match> contractedMatches = Collections.emptyList(); public Partitions( GraphPartitioner graphPartitioner, ElementGraph beginGraph, Map<ElementGraph, EnumMultiMap> annotatedSubGraphs ) { this( graphPartitioner, beginGraph, null, null, annotatedSubGraphs ); } public Partitions( GraphPartitioner graphPartitioner, ElementGraph beginGraph, ElementGraph contractedGraph, List<Match> contractedMatches, Map<ElementGraph, EnumMultiMap> annotatedSubGraphs ) { this.graphPartitioner = graphPartitioner; this.beginGraph = beginGraph; if( contractedGraph != null ) this.contractedGraph = contractedGraph; if( contractedMatches != null ) this.contractedMatches = contractedMatches; this.annotatedSubGraphs = annotatedSubGraphs; } public void setRulePartitioner( RulePartitioner rulePartitioner ) { this.rulePartitioner = rulePartitioner; } public String getRuleName() { if( rulePartitioner != null ) return rulePartitioner.getRuleName(); return "none"; } @Override public ElementGraph getBeginGraph() { return beginGraph; } @Override public ElementGraph getEndGraph() { return null; } public Map<ElementGraph, EnumMultiMap> getAnnotatedSubGraphs() { return annotatedSubGraphs; } public boolean hasSubGraphs() { return !annotatedSubGraphs.isEmpty(); } public boolean hasContractedMatches() { return !contractedMatches.isEmpty(); } public List<ElementGraph> getSubGraphs() { return new ArrayList<>( annotatedSubGraphs.keySet() ); } @Override public void writeDOTs( String path ) { int count = 0; beginGraph.writeDOT( new File( path, makeFileName( count++, "element-graph" ) ).toString() ); if( graphPartitioner instanceof ExpressionGraphPartitioner ) { ExpressionGraphPartitioner expressionGraphPartitioner = (ExpressionGraphPartitioner) graphPartitioner; ExpressionGraph contractionGraph = expressionGraphPartitioner.getContractionGraph(); if( contractionGraph != null ) contractionGraph.writeDOT( new File( path, makeFileName( count++, "contraction-graph", contractionGraph ) ).toString() ); ExpressionGraph expressionGraph = expressionGraphPartitioner.getExpressionGraph(); if( expressionGraph != null ) expressionGraph.writeDOT( new File( path, makeFileName( count++, "expression-graph", expressionGraph ) ).toString() ); } if( contractedGraph != null ) contractedGraph.writeDOT( new File( path, makeFileName( count++, "contracted-graph" ) ).toString() ); List<ElementGraph> subGraphs = getSubGraphs(); for( int i = 0; i < subGraphs.size(); i++ ) { ElementGraph subGraph = subGraphs.get( i ); new ElementMultiGraph( subGraph, annotatedSubGraphs.get( subGraph ) ).writeDOT( new File( path, makeFileName( count, i, "partition-result-sub-graph" ) ).toString() ); if( i < contractedMatches.size() ) contractedMatches.get( i ).getMatchedGraph().writeDOT( new File( path, makeFileName( count, i, "partition-contracted-graph" ) ).toString() ); } } private String makeFileName( int ordinal, String name ) { return String.format( "%02d-%s.dot", ordinal, name ); } private String makeFileName( int ordinal, String name, Object type ) { return String.format( "%02d-%s-%s.dot", ordinal, name, type.getClass().getSimpleName() ); } private String makeFileName( int order, int ordinal, String name ) { return String.format( "%02d-%04d-%s.dot", order, ordinal, name ); } }