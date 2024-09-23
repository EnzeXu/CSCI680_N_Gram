public class ExpressionSubGraphIterator implements SubGraphIterator { private final PlannerContext plannerContext; private final ElementGraph elementGraph; private ContractedTransformer contractedTransformer; private GraphFinder graphFinder; private Set<FlowElement> elementExcludes = createIdentitySet(); private ElementGraph contractedGraph; private Transformed<ElementGraph> contractedTransformed; private boolean firstOnly = false; private Match match; private List<Match> matches = new ArrayList<>(); int count = 0; public ExpressionSubGraphIterator( ExpressionGraph matchExpression, ElementGraph elementGraph ) { this( new PlannerContext(), matchExpression, elementGraph ); } public ExpressionSubGraphIterator( PlannerContext plannerContext, ExpressionGraph matchExpression, ElementGraph elementGraph ) { this( plannerContext, null, matchExpression, elementGraph ); } public ExpressionSubGraphIterator( PlannerContext plannerContext, ExpressionGraph contractionExpression, ExpressionGraph matchExpression, ElementGraph elementGraph ) { this( plannerContext, contractionExpression, matchExpression, false, elementGraph ); } public ExpressionSubGraphIterator( PlannerContext plannerContext, ExpressionGraph contractionExpression, ExpressionGraph matchExpression, ElementGraph elementGraph, Collection<FlowElement> elementExcludes ) { this( plannerContext, contractionExpression, matchExpression, false, elementGraph, elementExcludes ); } public ExpressionSubGraphIterator( PlannerContext plannerContext, ExpressionGraph contractionExpression, ExpressionGraph matchExpression, boolean firstOnly, ElementGraph elementGraph ) { this( plannerContext, contractionExpression, matchExpression, firstOnly, elementGraph, null ); } public ExpressionSubGraphIterator( PlannerContext plannerContext, ExpressionGraph contractionExpression, ExpressionGraph matchExpression, boolean firstOnly, ElementGraph elementGraph, Collection<FlowElement> elementExcludes ) { this.plannerContext = plannerContext; this.firstOnly = firstOnly; this.elementGraph = elementGraph; if( elementExcludes != null ) this.elementExcludes.addAll( elementExcludes ); if( contractionExpression != null ) contractedTransformer = new ContractedTransformer( contractionExpression ); else contractedGraph = elementGraph; graphFinder = new GraphFinder( matchExpression ); } @Override public ElementGraph getElementGraph() { return elementGraph; } public List<Match> getMatches() { return matches; } public ElementGraph getContractedGraph() { if( contractedGraph == null ) { contractedTransformed = contractedTransformer.transform( plannerContext, elementGraph ); contractedGraph = contractedTransformed.getEndGraph(); } return contractedGraph; } public Match getLastMatch() { if( matches.isEmpty() ) return null; return matches.get( count - 1 ); } @Override public EnumMultiMap getAnnotationMap( ElementAnnotation[] annotations ) { EnumMultiMap annotationsMap = new EnumMultiMap(); if( annotations.length == 0 ) return annotationsMap; Match match = getLastMatch(); for( ElementAnnotation annotation : annotations ) annotationsMap.addAll( annotation.getAnnotation(), match.getCapturedElements( annotation.getCapture() ) ); return annotationsMap; } @Override public boolean hasNext() { if( match == null ) { match = graphFinder.findMatchesOnPrimary( plannerContext, getContractedGraph(), firstOnly, elementExcludes ); if( match.foundMatch() ) { matches.add( match ); elementExcludes.addAll( match.getCapturedElements( ElementCapture.Primary ) ); count++; } } return match.foundMatch(); } @Override public ElementGraph next() { try { if( !hasNext() ) throw new NoSuchElementException(); ElementGraph contractedMatchedGraph = match.getMatchedGraph(); Set<FlowElement> excludes = getContractedGraph().vertexSetCopy(); excludes.removeAll( contractedMatchedGraph.vertexSet() ); return asSubGraph( elementGraph, contractedMatchedGraph, excludes ); } finally { match = null; } } @Override public void remove() { throw new UnsupportedOperationException(); } }