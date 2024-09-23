public class ExpressionGraphPartitioner extends GraphPartitioner { protected ExpressionGraph contractionGraph ; protected ExpressionGraph expressionGraph ; protected ElementAnnotation [ ] annotations = new ElementAnnotation [ 0 ] ; public ExpressionGraphPartitioner ( ExpressionGraph contractionGraph , ExpressionGraph expressionGraph , ElementAnnotation . . . annotations ) { this . contractionGraph = contractionGraph ; this . expressionGraph = expressionGraph ; this . annotations = annotations ; } public ExpressionGraph getContractionGraph ( ) { return contractionGraph ; } public ExpressionGraph getExpressionGraph ( ) { return expressionGraph ; } public ElementAnnotation [ ] getAnnotations ( ) { return annotations ; } public void setAnnotations ( ElementAnnotation [ ] annotations ) { this . annotations = annotations ; } @ Override public Partitions partition ( PlannerContext plannerContext , ElementGraph elementGraph , Collection < FlowElement > excludes ) { Map < ElementGraph , EnumMultiMap > annotatedSubGraphs = new LinkedHashMap < > ( ) ; ExpressionSubGraphIterator expressionIterator = new ExpressionSubGraphIterator ( plannerContext , contractionGraph , expressionGraph , elementGraph , excludes ) ; SubGraphIterator stepIterator = wrapIterator ( expressionIterator ) ; while ( stepIterator . hasNext ( ) ) { ElementGraph next = stepIterator . next ( ) ; EnumMultiMap annotationMap = stepIterator . getAnnotationMap ( annotations ) ; annotatedSubGraphs . put ( next , annotationMap ) ; } return new Partitions ( this , elementGraph , expressionIterator . getContractedGraph ( ) , expressionIterator . getMatches ( ) , annotatedSubGraphs ) ; } protected SubGraphIterator wrapIterator ( ExpressionSubGraphIterator expressionIterator ) { return expressionIterator ; } }