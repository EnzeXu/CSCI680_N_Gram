public class PlannerLevelElementExpression extends ElementExpression { private final Class<? extends PlannerLevel> plannerLevelClass; protected PlannerLevelElementExpression( ElementCapture capture, Class<? extends PlannerLevel> plannerLevelClass ) { super( capture ); this.plannerLevelClass = plannerLevelClass; } @Override public boolean applies( PlannerContext plannerContext, ElementGraph elementGraph, FlowElement flowElement ) { if( !( flowElement instanceof Operator ) ) return false; Operator operator = (Operator) flowElement; if( !operator.hasPlannerLevel() ) return false; PlannerLevel plannerLevel = plannerContext.getPlannerLevelFor( plannerLevelClass ); if( plannerLevel == null ) return false; if( !( (PlannedOperation) operator.getOperation() ).supportsPlannerLevel( plannerLevel ) ) return false; return operator.getPlannerLevel().isStricterThan( plannerLevel ); } }