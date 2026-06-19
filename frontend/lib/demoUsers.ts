export interface DemoAccount {
  username: string;
  password: string;
  role: string;
  label: string;
}

export const DEMO_ACCOUNTS: DemoAccount[] = [
  { username: "dr.mehta", password: "medibot123", role: "doctor", label: "Dr. Mehta — Doctor" },
  { username: "nurse.priya", password: "medibot123", role: "nurse", label: "Nurse Priya — Nurse" },
  { username: "billing.ravi", password: "medibot123", role: "billing_executive", label: "Ravi — Billing Executive" },
  { username: "tech.anand", password: "medibot123", role: "technician", label: "Anand — Technician" },
  { username: "admin.sys", password: "medibot123", role: "admin", label: "System Admin — Admin" },
];

export const ROLE_COLORS: Record<string, string> = {
  doctor: "#1E88E5",
  nurse: "#00897B",
  billing_executive: "#8E24AA",
  technician: "#F4511E",
  admin: "#3949AB",
};
