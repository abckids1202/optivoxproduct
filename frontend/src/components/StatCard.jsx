export default function StatCard({ label, value, detail, icon: Icon, tone = "neutral" }) {
  return (
    <section className={`stat-card stat-${tone}`}>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        {detail && <small>{detail}</small>}
      </div>
      {Icon && (
        <div className="stat-icon">
          <Icon size={20} />
        </div>
      )}
    </section>
  );
}
