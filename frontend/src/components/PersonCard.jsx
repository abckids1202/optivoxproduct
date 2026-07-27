import { UserCheck, UserPlus, UserX } from "lucide-react";

export default function PersonCard({ person }) {
  const Icon = person.type === "registered" ? UserCheck : person.type === "spoof" ? UserX : UserPlus;

  return (
    <article className={`person-card ${person.type}`}>
      <div className="person-icon">
        <Icon size={19} />
      </div>
      <div className="person-main">
        <div className="person-title">
          <strong>{person.label}</strong>
          <span>{Math.round(person.confidence * 100)}%</span>
        </div>
        <p>{person.note}</p>
        <div className="person-meta">
          <span>{person.attendance}</span>
          <span>{person.visibleFor}</span>
        </div>
      </div>
      {person.type === "unknown" && <button type="button">Register</button>}
    </article>
  );
}
